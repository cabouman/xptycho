"""The Reconstruction: what :meth:`~xptycho.PtychoModel.recon` returns."""
import csv
import os

import h5py
import numpy as np


class Reconstruction:
    """The result of a reconstruction: the two unknowns and the record of
    the run.

    A ``Reconstruction`` can be passed back to
    :meth:`~xptycho.PtychoModel.recon` as ``init`` to start another run
    from it, and :meth:`save` writes a folder that :meth:`load` reads.

    .. list-table::
       :header-rows: 1
       :widths: 18 30 52

       * - attribute
         - type
         - meaning
       * - ``object``
         - complex64 ``(rows, cols)``
         - the object image on the object grid
       * - ``probe``
         - complex64 ``(K, n, n)``
         - the probe modes the run used or estimated
       * - ``pixel_size``
         - float, meters
         - the size of an object pixel
       * - ``origin``
         - ``(row, col)``, meters
         - the position of the center of the first pixel of the grid
       * - ``coverage``
         - float32 ``(rows, cols)``
         - the accumulated probe weight; zero where no probe reached
       * - ``positions``
         - float64 ``(J, 2)``, meters
         - the positions the run used
       * - ``mode_energies``
         - float ``(K,)``
         - the share of the probe energy in each mode
       * - ``params``
         - list of dict
         - every parameter of the run with its value, units, and origin
       * - ``curves``
         - dict of lists
         - ``data_error`` per iteration
       * - ``iterations``
         - int
         - the iterations run
    """

    def __init__(self, object, probe, pixel_size, origin, coverage, positions, params, curves, iterations):
        self.object = object
        self.probe = probe
        self.pixel_size = pixel_size
        self.origin = tuple(origin)
        self.coverage = coverage
        self.positions = positions
        self.params = params
        self.curves = curves
        self.iterations = iterations

    @property
    def mode_energies(self):
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
        """A boolean mask of the object grid, True inside the rectangle
        spanned by the centers of the scan positions.  This is the region
        every probe position surrounds, where the object is best
        determined; comparisons with a truth are made inside it."""
        centers = (self.positions - np.asarray(self.origin)) / self.pixel_size
        first = np.floor(centers.min(axis=0)).astype(int)
        last = np.ceil(centers.max(axis=0)).astype(int)
        mask = np.zeros(self.object.shape, dtype=bool)
        mask[first[0]:last[0] + 1, first[1]:last[1] + 1] = True
        return mask

    def summary(self):
        """Return the parameter table as text, then the outcome of the run."""
        width = max(len(row['name']) for row in self.params)
        lines = ['Reconstruction', '  parameters:']
        for row in self.params:
            lines.append('    {:<{w}}  {}  {}  [{}]'.format(row['name'], row['value'], row['units'], row['origin'], w=width))
        lines.append('  iterations: {}'.format(self.iterations))
        if self.curves.get('data_error'):
            lines.append('  final data error: {:.6f}'.format(self.curves['data_error'][-1]))
        lines.append('  mode energies: ' + ', '.join('{:.4f}'.format(e) for e in self.mode_energies))
        return '\n'.join(lines)

    def show(self, directory=None, compare_to=None):
        """Plot the magnitude and phase of the object, the probe modes, and
        the data-error curve.

        Args:
            directory (str, optional): where the figures are saved.  None
                shows them without saving.
            compare_to (Truth, optional): a known object.  Adds the truth to
                the figure and prints the NRMSE of the object inside
                :meth:`scanned_region`.
        """
        from .view import show_reconstruction
        show_reconstruction(self, directory, compare_to)

    def save(self, directory):
        """Write one folder: ``summary.txt``, ``parameters.csv``, and
        ``recon.h5`` (the object, the probe, the positions, the coverage,
        and the curves, with the pixel size and origin as attributes)."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'summary.txt'), 'w') as f:
            f.write(self.summary() + '\n')
        with open(os.path.join(directory, 'parameters.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'value', 'units', 'origin'])
            writer.writeheader()
            writer.writerows(self.params)
        with h5py.File(os.path.join(directory, 'recon.h5'), 'w') as f:
            f.create_dataset('object', data=self.object)
            f.create_dataset('probe', data=self.probe)
            f.create_dataset('coverage', data=self.coverage)
            f.create_dataset('positions', data=self.positions)
            for name, values in self.curves.items():
                f.create_dataset('curves/' + name, data=np.asarray(values))
            f.attrs['pixel_size'] = self.pixel_size
            f.attrs['origin'] = self.origin
            f.attrs['iterations'] = self.iterations

    @classmethod
    def load(cls, directory):
        """Read a folder written by :meth:`save`."""
        with open(os.path.join(directory, 'parameters.csv'), newline='') as f:
            params = list(csv.DictReader(f))
        with h5py.File(os.path.join(directory, 'recon.h5'), 'r') as f:
            curves = {name: list(f['curves'][name][...]) for name in f.get('curves', {})}
            return cls(f['object'][...], f['probe'][...], float(f.attrs['pixel_size']),
                       tuple(f.attrs['origin']), f['coverage'][...], f['positions'][...],
                       params, curves, int(f.attrs['iterations']))
