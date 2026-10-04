"""The Sample: an object and the probe it was seen with."""
import h5py
import numpy as np

FORMAT_VERSION = 1
PARAMETER_COLUMNS = ('name', 'value', 'units', 'origin')


def open_group(file, group):
    """Return a new, empty group of an open HDF5 file, replacing the group
    if the file already has it.  The other groups of the file are kept."""
    if group in file:
        del file[group]
    file.attrs['format_version'] = FORMAT_VERSION
    return file.create_group(group)


class RunRecord:
    """How an estimated :class:`Sample` was made.

    Attributes:
        parameters (list of dict): every parameter of the run with its
            ``name``, ``value``, ``units``, and ``origin``.
        data_error (list of float): the data error after each iteration.
        iterations (int): the iterations run.
        positions (ndarray): float64 ``(J, 2)``, meters, the positions the
            run used.
        coverage (ndarray): float32 ``(rows, cols)``, the accumulated probe
            weight; zero where no probe reached.
    """

    def __init__(self, parameters, data_error, iterations, positions, coverage):
        self.parameters = parameters
        self.data_error = list(data_error)
        self.iterations = int(iterations)
        self.positions = positions
        self.coverage = coverage


class Sample:
    """An object and the probe it was seen with: the two unknowns of
    ptychography.

    A ground truth and a reconstruction are both a ``Sample``.
    :meth:`~xptycho.PtychoModel.simulate` takes one and
    :meth:`~xptycho.PtychoModel.recon` returns one.  A reconstruction also
    carries ``run``, the record of how it was made.

    Args:
        object (ndarray): complex ``(rows, cols)``, the transmittance image.
        probe (ndarray): complex ``(K, n, n)`` or ``(n, n)``, the probe modes.
        pixel_pitch (float): meters, the spacing of the pixels of both.
        origin (tuple of float, optional): ``(row, col)`` in meters, the
            position of the center of the first pixel of the object.
        name (str, optional): a label for summaries and figures.
        run (RunRecord, optional): the record of the run, for an estimate.

    In a file, a sample is the group ``/sample`` of an HDF5 file, with
    datasets ``object`` and ``probe`` and attributes ``pixel_pitch``,
    ``origin``, and ``name``.  The record of the run is the group ``/run``,
    with datasets ``coverage``, ``positions``, ``data_error``, and
    ``parameters`` and the attribute ``iterations``.  The same file may also
    hold a :class:`~xptycho.Scan` in ``/scan``.
    """

    def __init__(self, object, probe, pixel_pitch, origin=None, name=None, run=None):
        probe = np.asarray(probe, dtype=np.complex64)
        self.object = np.asarray(object, dtype=np.complex64)
        self.probe = probe[None] if probe.ndim == 2 else probe
        if self.object.ndim != 2 or self.probe.ndim != 3:
            raise ValueError('the object must be (rows, cols) and the probe (K, n, n) or (n, n); got {} and {}'.format(
                self.object.shape, probe.shape))
        self.pixel_pitch = float(pixel_pitch)
        self.origin = None if origin is None else tuple(float(v) for v in origin)
        self.name = name
        self.run = run

    @property
    def mode_energies(self):
        """The share of the probe energy in each mode, float ``(K,)``."""
        energy = (np.abs(self.probe) ** 2).sum(axis=(-2, -1))
        return energy / energy.sum()

    @property
    def phase(self):
        """The phase of the object in radians, float32."""
        return np.angle(self.object).astype(np.float32)

    @property
    def magnitude(self):
        """The magnitude of the object, float32."""
        return np.abs(self.object).astype(np.float32)

    def scanned_region(self):
        """A boolean mask of the object, True inside the rectangle spanned
        by the centers of the scan positions of the run.  This is the region
        every probe position surrounds, where the object is best determined;
        comparisons with a truth are made inside it.  Needs ``run``."""
        if self.run is None:
            raise ValueError('scanned_region needs the record of a run; this sample has none')
        centers = (self.run.positions - np.asarray(self.origin)) / self.pixel_pitch
        first = np.floor(centers.min(axis=0)).astype(int)
        last = np.ceil(centers.max(axis=0)).astype(int)
        mask = np.zeros(self.object.shape, dtype=bool)
        mask[first[0]:last[0] + 1, first[1]:last[1] + 1] = True
        return mask

    def summary(self):
        """Return a text summary: the sample, then the parameters and the
        outcome of the run when there is one."""
        lines = [
            'Sample{}'.format(' "{}"'.format(self.name) if self.name else ''),
            '  object:        {} x {} pixels'.format(*self.object.shape),
            '  probe:         {} mode{} of {} x {} pixels'.format(
                len(self.probe), '' if len(self.probe) == 1 else 's', *self.probe.shape[1:]),
            '  pixel pitch:   {:.6g} m'.format(self.pixel_pitch),
            '  mode energies: ' + ', '.join('{:.4f}'.format(e) for e in self.mode_energies),
        ]
        if self.run is not None:
            width = max(len(row['name']) for row in self.run.parameters)
            lines.append('  parameters of the run:')
            for row in self.run.parameters:
                lines.append('    {:<{w}}  {}  {}  [{}]'.format(row['name'], row['value'], row['units'], row['origin'], w=width))
            lines.append('  iterations: {}'.format(self.run.iterations))
            if self.run.data_error:
                lines.append('  final data error: {:.6f}'.format(self.run.data_error[-1]))
        return '\n'.join(lines)

    def save(self, path):
        """Write the sample to the group ``/sample`` of an HDF5 file, and
        the record of the run, when there is one, to ``/run``.  A file that
        exists keeps its other groups."""
        with h5py.File(path, 'a') as f:
            g = open_group(f, 'sample')
            g.create_dataset('object', data=self.object, compression='gzip', shuffle=True)
            g.create_dataset('probe', data=self.probe, compression='gzip', shuffle=True)
            g.attrs['pixel_pitch'] = self.pixel_pitch
            g.attrs['units'] = 'pixel_pitch and origin in m'
            if self.origin is not None:
                g.attrs['origin'] = self.origin
            if self.name:
                g.attrs['name'] = self.name
            if 'run' in f:
                del f['run']
            if self.run is not None:
                r = f.create_group('run')
                r.create_dataset('coverage', data=self.run.coverage)
                r.create_dataset('positions', data=self.run.positions)
                r.create_dataset('data_error', data=np.asarray(self.run.data_error, dtype=np.float64))
                table = [[str(row[c]) for c in PARAMETER_COLUMNS] for row in self.run.parameters]
                r.create_dataset('parameters', data=np.array(table, dtype=object), dtype=h5py.string_dtype())
                r['parameters'].attrs['columns'] = list(PARAMETER_COLUMNS)
                r.attrs['iterations'] = self.run.iterations

    @classmethod
    def load(cls, path):
        """Read the sample in the group ``/sample`` of an HDF5 file, with
        the record of the run when the file has ``/run``."""
        with h5py.File(path, 'r') as f:
            if 'sample' not in f:
                raise ValueError('{} holds no sample; its groups are: {}'.format(path, ', '.join(f) or 'none'))
            g = f['sample']
            run = None
            if 'run' in f:
                r = f['run']
                table = r['parameters'].asstr()[...]
                parameters = [dict(zip(PARAMETER_COLUMNS, row)) for row in table]
                run = RunRecord(parameters, r['data_error'][...].tolist(), r.attrs['iterations'],
                                r['positions'][...], r['coverage'][...])
            origin = g.attrs.get('origin')
            return cls(g['object'][...], g['probe'][...], float(g.attrs['pixel_pitch']),
                       None if origin is None else tuple(origin), g.attrs.get('name'), run)
