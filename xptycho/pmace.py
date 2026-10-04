"""The PMACE iteration, as on the Theory and Computing pages of the design.

One :class:`Run` carries out a reconstruction on the devices of a
:class:`~xptycho._sharding.Layout`.  Each device holds one block of
positions and computes over it; the two operations of the layout, sum to
owner and copy to window, form every sum over all the positions.  With
one device the same code runs with nothing to exchange.
"""
import numpy as np
import torch

from . import operators as op

MODE_WEIGHT_EXPONENT = 2      # mode weights are energies, ||d_k||**2
OUTLIER_THRESHOLD = 1.5       # on the modified z-score of a probe copy
OUTLIER_SCALE = 0.6745


def fresnel_propagate(field, wavelength, distance, pixel_size):
    """Propagate a square complex field a distance, by the Fresnel
    transfer function.  numpy, on the host; the field is probe-sized."""
    n = field.shape[0]
    f = np.fft.fftfreq(n, d=pixel_size)
    fx, fy = np.meshgrid(f, f, indexing='ij')
    transfer = np.exp(-1j * np.pi * wavelength * distance * (fx ** 2 + fy ** 2))
    return np.fft.ifft2(np.fft.fft2(field) * transfer)


class Run:
    """The state of one reconstruction and the steps of its iteration.

    Args:
        layout (Layout): the blocks, bands, and windows.
        starts (ndarray): int ``(J, 2)``, the first pixel of each patch.
        amplitudes (ndarray): float32 ``(J, n, n)``, the measured
            amplitudes in the scan's own order.
        probe (ndarray): complex ``(K, n, n)``, the probe modes: the known
            probe, or the starting estimate.
        init_object (ndarray): complex ``(rows, cols)``, the starting
            object.
        params (dict): the reconstruction parameters.
        estimate_probe (bool): estimate the probe with the object.
        batch_size (int): positions processed together on a device.
    """

    def __init__(self, layout, starts, amplitudes, probe, init_object, params,
                 estimate_probe, batch_size):
        self.layout = layout
        self.params = params
        self.estimate_probe = estimate_probe
        self.batch_size = batch_size
        self.num_positions = len(starts)
        self.size = amplitudes.shape[-1]
        self.num_pixels = layout.object_shape[0] * layout.object_shape[1]
        self.modes = torch.as_tensor(np.asarray(probe, dtype=np.complex64)).to(layout.devices[0])

        self.starts, self.y, self.v, self.s = [], [], [], []
        init_windows = layout.copy_to_window(layout.split(np.asarray(init_object, dtype=np.complex64)))
        for g, device in enumerate(layout.devices):
            block, window = layout.blocks[g], layout.windows[g]
            local = starts[block].copy()
            local[:, 0] -= window[0]
            self.starts.append(torch.as_tensor(local, dtype=torch.int64, device=device))
            self.y.append(torch.as_tensor(amplitudes[block], dtype=torch.float32, device=device))
            self.v.append(op.gather_patches(init_windows[g], self.starts[g], self.size))
            if estimate_probe:
                copies = self.modes.to(device)[:, None].repeat(1, len(block), 1, 1)
                self.s.append(copies)
        self._set_modes(self.modes)
        self.image = None           # bands of the reported image
        self.image_windows = None   # the same image over each device's window

    # ----------------------------------------------------------- the modes
    def _set_modes(self, modes):
        """Broadcast the modes and recompute what follows from them: the
        mode weights, the stable-division constants, and the coverage."""
        kappa = self.params['probe_weight_exponent']
        self.modes = modes
        self.d = self.layout.broadcast(modes)
        energy = torch.linalg.vector_norm(modes, dim=(-2, -1)) ** MODE_WEIGHT_EXPONENT
        self.mode_weight = self.layout.broadcast(energy / energy.sum())
        eps = op.rms_epsilon((modes.abs() ** 2).sum(dim=(-2, -1)), self.size ** 2)
        self.mode_eps = self.layout.broadcast(eps[:, None, None])
        self.patch_weight = [d.abs() ** kappa for d in self.d]

        # The coverage of each mode: its weight added at every position.
        parts = []
        for g, device in enumerate(self.layout.devices):
            rows = self.layout.windows[g][1] - self.layout.windows[g][0]
            part = torch.zeros((len(modes), rows, self.layout.object_shape[1]), dtype=torch.complex64, device=device)
            weight = self.patch_weight[g].to(torch.complex64)
            for k in range(len(modes)):
                op.scatter_patches(weight[k].expand(len(self.starts[g]), -1, -1), self.starts[g], part[k])
            parts.append(part)
        self.coverage = [band / (1 + 1e-6) for band in self.layout.sum_to_owner(parts)]
        squares = self.layout.sum_small([(c.abs() ** 2).sum(dim=(-2, -1)) for c in self.coverage])
        eps = op.rms_epsilon(squares, self.num_pixels)
        self.coverage_eps = self.layout.broadcast(eps[:, None, None])

    def _average(self, parts):
        """The weighted average image from the devices' partial sums, one
        per mode: sum to owner, divide by the coverage, weight the modes."""
        sums = self.layout.sum_to_owner(parts)
        bands = []
        for g in range(len(self.layout)):
            ratio = op.stable_divide(sums[g], self.coverage[g], self.coverage_eps[g])
            bands.append((self.mode_weight[g][:, None, None] * ratio).sum(dim=0))
        return bands

    def _batches(self, g):
        count = len(self.starts[g])
        for first in range(0, count, self.batch_size):
            yield slice(first, min(first + self.batch_size, count))

    def _new_parts(self):
        """A zero partial sum per mode over each device's window."""
        parts = []
        for g, device in enumerate(self.layout.devices):
            rows = self.layout.windows[g][1] - self.layout.windows[g][0]
            parts.append(torch.zeros((len(self.modes), rows, self.layout.object_shape[1]),
                                     dtype=torch.complex64, device=device))
        return parts

    def _add_weighted(self, part, g, patches, batch):
        for k in range(len(self.modes)):
            op.scatter_patches(self.patch_weight[g][k] * patches, self.starts[g][batch], part[k])

    # ------------------------------------------------------ the object pass
    def update_object(self):
        """Steps 1 and 2: fit every patch to its data, average, update
        every patch, and average again.  Sets the reported image.

        The update ``v + 2 rho (z - w)`` is done in place in two parts:
        ``2 rho w`` is subtracted when ``w`` is computed, and ``2 rho z``
        is added once the average is known, so ``w`` is never stored."""
        alpha, rho = self.params['object_data_fit'], self.params['relaxation']

        parts = self._new_parts()
        for g in range(len(self.layout)):
            d, weight, eps = self.d[g], self.mode_weight[g], self.mode_eps[g]
            for batch in self._batches(g):
                v, y = self.v[g][batch], self.y[g][batch]
                fields = op.fft2c(d[:, None] * v)                        # (K, B, n, n)
                intensity = (fields.abs() ** 2).sum(dim=0)
                scale = torch.sqrt(y ** 2 / (intensity + 1e-6))
                fitted = op.ifft2c(torch.polar(scale * fields.abs(), fields.angle()))
                fitted = op.stable_divide(fitted, d[:, None], eps[:, None])
                w = (1 - alpha) * v + alpha * (weight[:, None, None, None] * fitted).sum(dim=0)
                self._add_weighted(parts[g], g, 2 * w - v, batch)
                self.v[g][batch] = v - 2 * rho * w
        consensus = self.layout.copy_to_window(self._average(parts))

        parts = self._new_parts()
        for g in range(len(self.layout)):
            for batch in self._batches(g):
                z = op.gather_patches(consensus[g], self.starts[g][batch], self.size)
                v = self.v[g][batch] + 2 * rho * z
                self.v[g][batch] = v
                self._add_weighted(parts[g], g, v, batch)
        self.image = self._average(parts)
        self.image_windows = self.layout.copy_to_window(self.image)

    def _image_patches(self, g, batch):
        return op.gather_patches(self.image_windows[g], self.starts[g][batch], self.size)

    def _patch_eps(self):
        """The stable-division constant for a division by the image
        patches: from the sum over all the positions."""
        sums = []
        for g in range(len(self.layout)):
            total = torch.zeros((), dtype=torch.float32, device=self.layout.devices[g])
            for batch in self._batches(g):
                total += (self._image_patches(g, batch).abs() ** 2).sum()
            sums.append(total)
        total = self.layout.sum_small(sums)
        return self.layout.broadcast(op.rms_epsilon(total, self.num_positions * self.size ** 2))

    # ------------------------------------------------------- the probe pass
    def update_probe(self):
        """Steps 4 to 6: fit every probe copy to its data, average, update
        every copy, take the new modes, and replace outlying copies.  The
        update is done in place in two parts, as in :meth:`update_object`."""
        alpha, rho = self.params['probe_data_fit'], self.params['relaxation']
        num_modes, J = len(self.modes), self.num_positions
        patch_eps = self._patch_eps()

        sums = []
        for g in range(len(self.layout)):
            d = self.d[g]
            total = torch.zeros_like(d)
            for batch in self._batches(g):
                p, y, s = self._image_patches(g, batch), self.y[g][batch], self.s[g][:, batch]
                fields = op.fft2c(d[:, None] * p)
                intensity = (fields.abs() ** 2).sum(dim=0)
                amplitude = torch.sqrt(y ** 2 / (intensity + 1e-6)) * fields.abs()
                phase = op.fft2c(s * p).angle()
                fitted = op.stable_divide(op.ifft2c(torch.polar(amplitude, phase)), p, patch_eps[g])
                r = (1 - alpha) * s + alpha * fitted
                total += (2 * r - s).sum(dim=1)
                self.s[g][:, batch] = s - 2 * rho * r
            sums.append(total)
        consensus = self.layout.broadcast(self.layout.sum_small(sums) / J)

        sums = []
        for g in range(len(self.layout)):
            self.s[g] += 2 * rho * consensus[g][:, None]
            sums.append(self.s[g].sum(dim=1))
        modes = self.layout.sum_small(sums) / J
        d = self.layout.broadcast(modes)

        sums = [(self.s[g] - d[g][:, None]).abs().sum(dim=1) for g in range(len(self.layout))]
        deviation = self.layout.sum_small(sums) / J
        deviation = torch.where(deviation == 0, torch.full_like(deviation, 1e-6), deviation)
        deviation = self.layout.broadcast(deviation)
        for g in range(len(self.layout)):
            center = d[g][:, None].expand_as(self.s[g])
            score = OUTLIER_SCALE * (self.s[g] - center).abs() / deviation[g][:, None]
            self.s[g] = torch.where(score > OUTLIER_THRESHOLD, center, self.s[g])
        self._set_modes(modes)

    def add_mode(self, wavelength, distance, pixel_size):
        """Step 3: add a probe mode from the intensity the current modes do
        not explain, and rescale every mode and every copy."""
        fraction = self.params['mode_energy_fraction']
        patch_eps = self._patch_eps()
        sums = []
        for g in range(len(self.layout)):
            d = self.d[g]
            total = torch.zeros((self.size, self.size), dtype=torch.complex64, device=d.device)
            for batch in self._batches(g):
                p, y = self._image_patches(g, batch), self.y[g][batch]
                intensity = (op.fft2c(d[:, None] * p).abs() ** 2).sum(dim=0)
                residual = torch.sqrt(torch.clamp(y ** 2 - intensity, min=0))
                total += op.stable_divide(op.ifft2c(residual.to(torch.complex64)), p, patch_eps[g]).sum(dim=0)
            sums.append(total)
        mean = (self.layout.sum_small(sums) / self.num_positions).cpu().numpy()
        new = fresnel_propagate(mean, wavelength, distance, pixel_size)
        energy = float((self.modes.abs() ** 2).sum())
        new = new * (np.sqrt(fraction * energy) / np.linalg.norm(new))
        new = torch.as_tensor(new.astype(np.complex64), device=self.modes.device)

        scale = 1 / np.sqrt(1 + fraction)
        modes = torch.cat([self.modes, new[None]]) * scale
        for g, device in enumerate(self.layout.devices):
            copies = new.to(device)[None, None].repeat(1, len(self.starts[g]), 1, 1)
            self.s[g] = torch.cat([self.s[g], copies]) * scale
        self._set_modes(modes)

    # ------------------------------------------------------------ reporting
    def data_error(self):
        """Step 7: the normalized root mean square difference between the
        measured amplitudes and those the reported image and the modes
        predict."""
        errors, norms = [], []
        for g in range(len(self.layout)):
            d = self.d[g]
            for batch in self._batches(g):
                p, y = self._image_patches(g, batch), self.y[g][batch]
                predicted = torch.sqrt((op.fft2c(d[:, None] * p).abs() ** 2).sum(dim=0))
                errors.append(((predicted - y) ** 2).sum())
                norms.append((y ** 2).sum())
        # The batch sums are added in double precision on the host.
        error = sum(float(e) for e in torch.stack([e.cpu() for e in errors]).double())
        norm = sum(float(n) for n in torch.stack([n.cpu() for n in norms]).double())
        return (error / norm) ** 0.5

    def object(self):
        """The reported image on the host, complex64 ``(rows, cols)``."""
        return self.layout.gather(self.image)

    def probe(self):
        """The probe modes on the host, complex64 ``(K, n, n)``."""
        return self.modes.cpu().numpy()

    def coverage_map(self):
        """The accumulated probe weight on the host, float32
        ``(rows, cols)``: zero where no probe reached."""
        bands = [(self.mode_weight[g][:, None, None] * self.coverage[g].real).sum(dim=0)
                 for g in range(len(self.layout))]
        return self.layout.gather(bands).astype(np.float32)
