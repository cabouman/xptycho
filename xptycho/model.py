"""The PtychoModel: the parameters and the forward model."""
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from . import operators as op
from ._sharding import Layout
from .pmace import Run, fresnel_propagate
from .sample import RunRecord, Sample
from .scan import HC_KEV_M, Scan, energy_to_wavelength

# name: (default, units)
RECON_DEFAULTS = {
    'object_data_fit': (0.6, ''),
    'probe_data_fit': (0.6, ''),
    'probe_weight_exponent': (1.25, ''),
    'relaxation': (0.5, ''),
    'mode_schedule': ((), 'iterations'),
    'mode_energy_fraction': (0.05, ''),
    'orthogonalize_modes': (False, ''),
    'initial_probe_distance': (None, 'm'),
    'object_shape': (None, 'pixels'),
    'object_origin': (None, 'm'),
    'batch_size': (None, 'positions'),
}
INITIAL_FILTER_SIGMA = 1.0     # pixels; smooths the starting object and probe


class PtychoModel:
    """The far-field forward model of a ptychographic scan, with the
    parameters of a reconstruction.

    A model holds three kinds of parameters and no image or probe.  The
    forward-model parameters are given here and fix the forward model.  The
    reconstruction parameters are set and read by name with
    :meth:`set_params` and :meth:`get_params`.  The devices are chosen with
    :meth:`configure_devices`.  The object and the probe are arguments of
    :meth:`forward`, :meth:`simulate`, and :meth:`recon`, and come back
    from :meth:`recon` in a :class:`~xptycho.Sample`.

    Args:
        wavelength (float, optional): meters.  Give this or ``energy``.
        energy (float, optional): keV.
        det_distance (float): meters, object to detector.
        det_pixel_pitch (float): meters, the detector pixel pitch.
        frame_size (int): the frames are ``frame_size`` square; even.
        probe_positions (ndarray): ``(J, 2)``, the row and column of the center
            of each probe position on the object, in meters.
        num_probe_modes (int, optional): the number of probe modes.  Defaults
            to 1.

    Example:
        .. code-block:: python

            model = xptycho.PtychoModel.from_scan(scan, num_probe_modes=2)
            model.set_params(object_data_fit=0.5, mode_schedule=[20])
            recon = model.recon(scan, iterations=200)
    """

    def __init__(self, *, wavelength=None, energy=None, det_distance, det_pixel_pitch,
                 frame_size, probe_positions, num_probe_modes=1, _origin='given'):
        if (wavelength is None) == (energy is None):
            raise ValueError('give exactly one of wavelength (meters) and energy (keV)')
        if frame_size % 2:
            raise ValueError('frame_size must be even; got {}'.format(frame_size))
        probe_positions = np.asarray(probe_positions, dtype=np.float64)
        if probe_positions.ndim != 2 or probe_positions.shape[1] != 2:
            raise ValueError('probe_positions must have shape (J, 2); got {}'.format(probe_positions.shape))
        self.wavelength = float(wavelength) if wavelength is not None else energy_to_wavelength(energy)
        self.det_distance = float(det_distance)
        self.det_pixel_pitch = float(det_pixel_pitch)
        self.frame_size = int(frame_size)
        self.probe_positions = probe_positions
        self.num_probe_modes = int(num_probe_modes)
        self._fmodel_origin = _origin
        self._recon = {name: default for name, (default, _) in RECON_DEFAULTS.items()}
        self._given = set()
        self._devices = None

    @classmethod
    def from_scan(cls, scan, num_probe_modes=1, **overrides):
        """Build a model from the facts a scan recorded.  Any constructor
        argument passed here overrides the scan's value."""
        facts = dict(wavelength=scan.wavelength, det_distance=scan.det_distance,
                     det_pixel_pitch=scan.det_pixel_pitch, frame_size=scan.frame_size,
                     probe_positions=scan.probe_positions)
        if 'energy' in overrides:
            facts.pop('wavelength')
        facts.update(overrides)
        return cls(num_probe_modes=num_probe_modes, _origin='given' if overrides else 'file', **facts)

    # ------------------------------------------------------------ parameters
    @property
    def sample_pixel_pitch(self):
        """The object pixel pitch in meters: wavelength times
        detector distance over frame size times detector pitch."""
        return self.wavelength * self.det_distance / (self.frame_size * self.det_pixel_pitch)

    def set_params(self, **params):
        """Set reconstruction parameters by name.  ``probe_positions`` may also be
        set, to replace the scan positions with refined ones."""
        for name, value in params.items():
            if name == 'probe_positions':
                value = np.asarray(value, dtype=np.float64)
                if value.shape != self.probe_positions.shape:
                    raise ValueError('probe_positions must have shape {}; got {}'.format(self.probe_positions.shape, value.shape))
                self.probe_positions = value
                self._fmodel_origin = 'given'
            elif name in self._recon:
                self._recon[name] = value
                self._given.add(name)
            else:
                raise ValueError('"{}" is not a reconstruction parameter; the parameters are: {}'.format(
                    name, ', '.join(self._recon)))

    def get_params(self, names):
        """Return one parameter, or a list for a list of names."""
        def one(name):
            if name in self._recon:
                return self._recon[name]
            if name in ('wavelength', 'det_distance', 'det_pixel_pitch', 'frame_size', 'probe_positions',
                        'num_probe_modes', 'sample_pixel_pitch'):
                return getattr(self, name)
            raise ValueError('"{}" is not a parameter'.format(name))
        return one(names) if isinstance(names, str) else [one(n) for n in names]

    def parameters(self):
        """Return every parameter as a row with its value, units, and
        origin (given, file, derived, or default)."""
        shape, origin = self._grid()
        rows = [
            ('wavelength', self.wavelength, 'm', self._fmodel_origin),
            ('energy', HC_KEV_M / self.wavelength, 'keV', 'derived'),
            ('det_distance', self.det_distance, 'm', self._fmodel_origin),
            ('det_pixel_pitch', self.det_pixel_pitch, 'm', self._fmodel_origin),
            ('frame_size', self.frame_size, 'pixels', self._fmodel_origin),
            ('probe_positions', '{} positions'.format(len(self.probe_positions)), 'm', self._fmodel_origin),
            ('num_probe_modes', self.num_probe_modes, '', 'given'),
            ('sample_pixel_pitch', self.sample_pixel_pitch, 'm', 'derived'),
        ]
        for name, (_, units) in RECON_DEFAULTS.items():
            value, source = self._recon[name], 'given' if name in self._given else 'default'
            if name == 'object_shape' and value is None:
                value, source = shape, 'derived'
            if name == 'object_origin' and value is None:
                value, source = origin, 'derived'
            rows.append((name, value, units, source))
        rows.append(('devices', ', '.join(str(d) for d in self._device_list()), '',
                     'given' if self._devices else 'default'))
        return [dict(name=n, value=v, units=u, origin=o) for n, v, u, o in rows]

    def print_params(self):
        """Print every parameter with its value, units, and origin."""
        rows = self.parameters()
        width = max(len(row['name']) for row in rows)
        for row in rows:
            print('{:<{w}}  {}  {}  [{}]'.format(row['name'], row['value'], row['units'], row['origin'], w=width))

    # ------------------------------------------------------- the object grid
    def _grid(self):
        """The object grid as ``(shape, origin)``.  With neither given, the
        smallest grid that holds every patch.  With only the shape given,
        a grid of that shape centered on position zero.  The origin is the
        position, in meters, of the center of the grid's first pixel."""
        shape, origin = self._recon['object_shape'], self._recon['object_origin']
        if origin is None and shape is not None:
            # A given shape with no origin: the grid is centered on position zero.
            origin = tuple(-(n // 2) * self.sample_pixel_pitch for n in shape)
        if origin is None:
            first = np.round(self.probe_positions / self.sample_pixel_pitch).astype(np.int64) - self.frame_size // 2
            origin = tuple(float(v) for v in first.min(axis=0) * self.sample_pixel_pitch)
        if shape is None:
            starts = self._starts(origin)
            shape = tuple(int(v) for v in starts.max(axis=0) + self.frame_size)
        return tuple(shape), tuple(origin)

    def _starts(self, origin):
        """The row and column of the first pixel of each patch on the grid."""
        centers = np.round((self.probe_positions - np.asarray(origin)) / self.sample_pixel_pitch).astype(np.int64)
        return centers - self.frame_size // 2

    def _layout(self):
        shape, origin = self._grid()
        starts = self._starts(origin)
        if starts.min() < 0 or (starts + self.frame_size > np.asarray(shape)).any():
            raise ValueError('a patch reaches outside the object grid of shape {} with origin {} m'.format(shape, origin))
        return Layout(self._device_list(), starts, self.frame_size, shape), starts

    # ---------------------------------------------------------------- devices
    def configure_devices(self, num_devices=1, devices=None):
        """Choose the devices a reconstruction runs on.

        Without a call the model chooses automatically: every CUDA GPU,
        else Apple's GPU (mps), else the CPU.  With several devices, the positions and the object
        image are divided among them; see the Computing page of the design.

        Args:
            num_devices (int, optional): the number of CUDA GPUs to use.
                1 (the default) uses the first GPU; with no CUDA GPU,
                Apple's GPU or the CPU.
            devices (list, optional): the devices, for example
                ``['cuda:0', 'cuda:2']`` or ``['cpu']``.  Overrides
                ``num_devices``.
        """
        if devices is None:
            available = torch.cuda.device_count()
            if num_devices > 1 and num_devices > available:
                raise ValueError('{} devices were asked for but this node has {} CUDA GPUs'.format(num_devices, available))
            if available:
                devices = ['cuda:{}'.format(i) for i in range(num_devices)]
            else:
                devices = ['mps' if torch.backends.mps.is_available() else 'cpu']
        self._devices = [torch.device(d) for d in devices]

    def _device_list(self):
        if self._devices is not None:
            return self._devices
        count = torch.cuda.device_count()
        if count:
            return [torch.device('cuda:{}'.format(i)) for i in range(count)]
        return [torch.device('mps' if torch.backends.mps.is_available() else 'cpu')]

    def _batch_size(self, device):
        if self._recon['batch_size'] is not None:
            return int(self._recon['batch_size'])
        if device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(device)
            return max(1, int(0.8 * free / (110 * self.frame_size ** 2)))
        return max(1, 2 ** 24 // self.frame_size ** 2)      # about 2 GB of temporary arrays

    # ------------------------------------------------------ the forward model
    def _check_probe(self, probe):
        probe = np.asarray(probe, dtype=np.complex64)
        if probe.ndim == 2:
            probe = probe[None]
        if probe.shape[1:] != (self.frame_size, self.frame_size):
            raise ValueError('the probe must be {0} x {0} pixels; got {1}'.format(self.frame_size, probe.shape))
        return probe

    def forward(self, x, d):
        """The forward model: the amplitudes the object ``x`` and the probe
        ``d`` give at every scan position, with no noise.

        For every position, the patch of ``x`` is multiplied by each probe
        mode and transformed to the far field, and the modes are summed in
        intensity.

        Args:
            x (ndarray): complex ``(rows, cols)``, the object on the object
                grid.
            d (ndarray): complex ``(n, n)`` or ``(K, n, n)``, the probe.

        Returns:
            ndarray: float32 ``(J, n, n)``.
        """
        layout, starts = self._layout()
        x = np.asarray(x, dtype=np.complex64)
        if x.shape != layout.object_shape:
            raise ValueError('the object must have the shape of the object grid, {}; got {}'.format(layout.object_shape, x.shape))
        device = layout.devices[0]
        image = torch.as_tensor(x, device=device)
        modes = torch.as_tensor(self._check_probe(d), device=device)
        starts = torch.as_tensor(starts, dtype=torch.int64, device=device)
        out = np.empty((len(starts), self.frame_size, self.frame_size), dtype=np.float32)
        step = self._batch_size(device)
        for first in range(0, len(starts), step):
            patches = op.gather_patches(image, starts[first:first + step], self.frame_size)
            intensity = (op.fft2c(modes[:, None] * patches).abs() ** 2).sum(dim=0)
            out[first:first + step] = torch.sqrt(intensity).cpu().numpy()
        return out

    def simulate(self, sample, peak_photons=None, dark_rate=0.0, seed=0):
        """Simulate a scan of a known sample.

        Args:
            sample (Sample): the object and the probe.  Its object must have
                the shape of the object grid and its pixel pitch must equal
                the model's.
            peak_photons (float, optional): the counts at the brightest
                detector pixel.  Poisson counts are drawn from the
                intensities scaled to this peak.  None returns the exact
                intensities.
            dark_rate (float, optional): the mean dark counts per pixel,
                added as a second Poisson term.  Defaults to 0.
            seed (int, optional): the random seed of the counts.

        Returns:
            Scan: with this model's positions and instrument facts.
        """
        if not np.isclose(sample.pixel_pitch, self.sample_pixel_pitch, rtol=1e-6):
            raise ValueError('the sample has pixel pitch {:.6g} m but the model has {:.6g} m'.format(
                sample.pixel_pitch, self.sample_pixel_pitch))
        intensity = self.forward(sample.object, sample.probe) ** 2
        if peak_photons is not None:
            rng = np.random.default_rng(seed)
            frames = rng.poisson(intensity * (peak_photons / intensity.max()))
            if dark_rate:
                frames = frames + rng.poisson(dark_rate, size=frames.shape)
            intensity = frames.astype(np.float32)
        return Scan(intensity, self.probe_positions, wavelength=self.wavelength,
                    det_distance=self.det_distance, det_pixel_pitch=self.det_pixel_pitch)

    def data_error(self, scan, x, d):
        """The normalized root mean square difference between the scan's
        amplitudes and :meth:`forward` of ``x`` and ``d``."""
        y = self._amplitudes(scan)
        predicted = self.forward(x, d)
        return float(np.sqrt(((predicted - y) ** 2).sum(dtype=np.float64) / (y ** 2).sum(dtype=np.float64)))

    def _amplitudes(self, scan):
        if scan.frame_size != self.frame_size or scan.num_frames != len(self.probe_positions):
            raise ValueError('the scan has {} frames of size {} but the model has {} positions and frame size {}'.format(
                scan.num_frames, scan.frame_size, len(self.probe_positions), self.frame_size))
        return scan.amplitudes()

    def refine_probe_positions(self, scan, x, d, max_shift=1):
        """Find, for each position, the whole-pixel shift at which the
        forward model best matches the data.  Changes nothing: accept the
        result with ``set_params(probe_positions=...)``.

        Every shift of up to ``max_shift`` pixels along each axis is tried,
        with the object ``x`` and the probe ``d`` held fixed.  A shift that
        would take the patch outside the object grid is not tried.

        Args:
            scan (Scan): the measurement.
            x, d: the current object and probe, as in :meth:`forward`.
            max_shift (int, optional): pixels.  Defaults to 1.

        Returns:
            tuple: ``(probe_positions, misfit)``.  ``probe_positions`` is ``(J, 2)`` in
            meters.  ``misfit`` is ``(J,)``: the fraction by which the best
            shift lowers the difference from the data, 0 where the current
            position is already the best.
        """
        layout, starts = self._layout()
        y = torch.as_tensor(self._amplitudes(scan))
        device = layout.devices[0]
        image = torch.as_tensor(np.asarray(x, dtype=np.complex64), device=device)
        modes = torch.as_tensor(self._check_probe(d), device=device)
        starts = torch.as_tensor(starts, dtype=torch.int64, device=device)
        limit = torch.tensor(layout.object_shape, device=device) - self.frame_size
        shifts = [(r, c) for r in range(-max_shift, max_shift + 1) for c in range(-max_shift, max_shift + 1)]
        errors = torch.full((len(shifts), len(starts)), float('inf'), device=device)
        step = self._batch_size(device)
        for first in range(0, len(starts), step):
            batch = slice(first, first + step)
            measured = y[batch].to(device)
            for i, shift in enumerate(shifts):
                moved = starts[batch] + torch.tensor(shift, device=device)
                inside = ((moved >= 0) & (moved <= limit)).all(dim=1)
                patches = op.gather_patches(image, torch.where(inside[:, None], moved, starts[batch]), self.frame_size)
                predicted = torch.sqrt((op.fft2c(modes[:, None] * patches).abs() ** 2).sum(dim=0))
                error = torch.linalg.vector_norm(predicted - measured, dim=(-2, -1))
                errors[i, batch] = torch.where(inside, error, torch.full_like(error, float('inf')))
        best = errors.argmin(dim=0)
        current = errors[shifts.index((0, 0))]
        gain = current - errors.min(dim=0).values
        misfit = torch.where(current > 0, gain / current, torch.zeros_like(gain)).cpu().numpy()
        shift = torch.tensor(shifts, device=device)[best].cpu().numpy()
        return self.probe_positions + shift * self.sample_pixel_pitch, misfit

    # ------------------------------------------------------------ the start
    def initial_probe(self, scan):
        """The starting probe a reconstruction uses when none is given: the
        mean over positions of the back-transformed amplitudes, Fresnel
        propagated by ``initial_probe_distance`` when it is set, then
        smoothed.  One mode, complex64 ``(1, n, n)``."""
        y = torch.as_tensor(self._amplitudes(scan))
        total = torch.zeros((self.frame_size, self.frame_size), dtype=torch.complex64)
        for first in range(0, len(y), 64):
            total += op.ifft2c(y[first:first + 64].to(torch.complex64)).sum(dim=0)
        probe = (total / len(y)).numpy() / (1 + 1e-6)
        distance = self._recon['initial_probe_distance']
        if distance is not None:
            probe = fresnel_propagate(probe, self.wavelength, distance, self.sample_pixel_pitch)
        probe = (gaussian_filter(probe.real, INITIAL_FILTER_SIGMA)
                 + 1j * gaussian_filter(probe.imag, INITIAL_FILTER_SIGMA))
        return probe.astype(np.complex64)[None]

    def initial_object(self, scan, d):
        """The starting object a reconstruction uses when none is given: a
        constant patch per position, scaled by its data, averaged over the
        grid, uncovered pixels set to the median, then smoothed.  Real
        valued, complex64 ``(rows, cols)``."""
        layout, starts = self._layout()
        y = self._amplitudes(scan)
        probe = self._check_probe(d)
        scale = np.sqrt(np.linalg.norm(y, axis=(-2, -1)) / np.linalg.norm(probe))
        total = np.zeros(layout.object_shape, dtype=np.complex64)
        count = np.zeros(layout.object_shape, dtype=np.complex64)
        n = self.frame_size
        for (row, col), value in zip(starts, scale):
            total[row:row + n, col:col + n] += value
            count[row:row + n, col:col + n] += 1
        count = count / (1 + 1e-6)
        eps = 1e-6 * np.sqrt((np.abs(count) ** 2).sum() / count.size)
        image = total * np.conj(count) / (np.abs(count) ** 2 + eps)
        image[image == 0] = np.median(image)
        return gaussian_filter(np.abs(image), INITIAL_FILTER_SIGMA).astype(np.complex64)

    # -------------------------------------------------------- reconstruction
    def recon(self, scan, probe=None, init_probe=None, init=None, iterations=100, verbose=1):
        """Reconstruct the object, and the probe when it is not given.

        Args:
            scan (Scan): the measurement.  Not modified.
            probe (ndarray, optional): the known probe, held fixed.  It
                must have all ``num_probe_modes`` modes.  None (the default)
                means the probe is estimated.
            init_probe (ndarray, optional): the starting probe when the
                probe is estimated.  It may have fewer modes than
                ``num_probe_modes``.  Defaults to :meth:`initial_probe`.
            init (Sample, optional): start from this sample: its object,
                and its probe when the probe is estimated and ``init_probe``
                is not given.
            iterations (int, optional): iterations to run.  Defaults to 100.
            verbose (int, optional): 1 (the default) prints one line per
                iteration; 0 prints nothing.

        Returns:
            Sample: the object, the probe, and ``run``, the record of the run.
        """
        start_time = time.time()
        y = self._amplitudes(scan)
        estimate = probe is None
        if not estimate:
            modes = self._check_probe(probe)
            if len(modes) != self.num_probe_modes:
                raise ValueError('the given probe has {} modes but the model has num_probe_modes={}'.format(len(modes), self.num_probe_modes))
        elif init_probe is not None:
            modes = self._check_probe(init_probe)
        elif init is not None:
            modes = self._check_probe(init.probe)
        else:
            modes = self.initial_probe(scan)

        schedule = sorted(int(i) for i in self._recon['mode_schedule'])
        if estimate:
            schedule = schedule[:self.num_probe_modes - len(modes)]
            if len(modes) + len(schedule) != self.num_probe_modes:
                raise ValueError('the probe starts with {} modes and must reach num_probe_modes={}, so mode_schedule needs {} '
                                 'iterations; it is {}'.format(
                                     len(modes), self.num_probe_modes, self.num_probe_modes - len(modes),
                                     list(self._recon['mode_schedule'])))
            if schedule and self._recon['initial_probe_distance'] is None:
                raise ValueError('adding a probe mode needs initial_probe_distance to be set')

        if init is not None:
            start_object = np.asarray(init.object if hasattr(init, 'object') else init, dtype=np.complex64)
        else:
            start_object = self.initial_object(scan, modes)

        layout, starts = self._layout()
        if start_object.shape != layout.object_shape:
            raise ValueError('the starting object must have the shape of the object grid, {}; got {}'.format(
                layout.object_shape, start_object.shape))
        batch = min(self._batch_size(d) for d in layout.devices)
        run = Run(layout, starts, y, modes, start_object, dict(self._recon), estimate, batch)
        if verbose:
            print('devices: {}   batch {}   positions {}   object {} x {}'.format(
                ' '.join(str(d) for d in layout.devices), batch, len(starts), *layout.object_shape))

        errors = []
        for iteration in range(1, iterations + 1):
            run.update_object()         # the most important line: one PMACE update of the object
            if estimate:
                if iteration in schedule:
                    run.add_mode(self.wavelength, self._recon['initial_probe_distance'], self.sample_pixel_pitch)
                    if self._recon['orthogonalize_modes']:
                        run.orthogonalize_modes()
                    if verbose:
                        print('iteration {}: mode {} added, {:g}% of the energy'.format(
                            iteration, len(run.modes), 100 * self._recon['mode_energy_fraction']))
                run.update_probe()
            errors.append(run.data_error())
            if verbose:
                print('iteration {:4d}   data error {:.6f}   modes {}'.format(iteration, errors[-1], len(run.modes)))

        params = self.parameters()
        params.append(dict(name='iterations', value=iterations, units='', origin='given'))
        params.append(dict(name='probe', value='estimated' if estimate else 'given', units='', origin='given'))
        params.append(dict(name='seconds', value=round(time.time() - start_time, 3), units='s', origin='derived'))
        _, origin = self._grid()
        record = RunRecord(params, errors, iterations, self.probe_positions.copy(), run.coverage_map())
        return Sample(run.object(), run.probe(), self.sample_pixel_pitch, origin, scan.name, record)
