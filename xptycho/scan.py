"""The Scan: the measurement."""
import h5py
import numpy as np

from .sample import open_group

HC_KEV_M = 1.2398419843320026e-9     # Planck constant times light speed, keV m


def energy_to_wavelength(energy):
    """Photon energy in keV to wavelength in meters."""
    return HC_KEV_M / energy


class Scan:
    """A ptychographic measurement: the diffraction frames, the scan
    positions, and the instrument facts.

    A ``Scan`` is never modified by a reconstruction.  Its frames are
    intensities: detector counts, with any dark subtraction and cropping
    already done by :func:`~xptycho.preprocess`.  The reconstruction uses
    their square root.

    Args:
        frames (ndarray): intensities, shape ``(num_frames, size, size)``.
        probe_positions (ndarray): ``(num_frames, 2)``, the row and column of
            the center of each probe position on the object, in meters.
        wavelength (float, optional): meters.  Give this or ``energy``.
        energy (float, optional): keV.
        det_distance (float): meters, object to detector.
        det_pixel_pitch (float): meters, the detector pixel pitch after any
            binning.
        name (str, optional): a label for summaries.

    In a file, a scan is the group ``/scan`` of an HDF5 file, with datasets
    ``frames`` and ``probe_positions`` and attributes ``wavelength``,
    ``det_distance``, ``det_pixel_pitch``, and ``name``.  The same file
    may also hold a :class:`~xptycho.Sample` in ``/sample``.
    """

    def __init__(self, frames, probe_positions, *, wavelength=None, energy=None,
                 det_distance, det_pixel_pitch, name=None):
        frames = np.asarray(frames)
        probe_positions = np.asarray(probe_positions, dtype=np.float64)
        if frames.ndim != 3 or frames.shape[1] != frames.shape[2]:
            raise ValueError('frames must have shape (num_frames, size, size); got {}'.format(frames.shape))
        if probe_positions.shape != (frames.shape[0], 2):
            raise ValueError('probe_positions must have shape ({}, 2); got {}'.format(frames.shape[0], probe_positions.shape))
        if (wavelength is None) == (energy is None):
            raise ValueError('give exactly one of wavelength (meters) and energy (keV)')
        self.frames = frames
        self.probe_positions = probe_positions
        self.wavelength = float(wavelength) if wavelength is not None else energy_to_wavelength(energy)
        self.det_distance = float(det_distance)
        self.det_pixel_pitch = float(det_pixel_pitch)
        self.name = name

    @property
    def num_frames(self):
        """The number of scan positions."""
        return self.frames.shape[0]

    @property
    def frame_size(self):
        """The frames are ``frame_size`` by ``frame_size`` pixels."""
        return self.frames.shape[1]

    def amplitudes(self):
        """The square root of the frames, float32; negative values are
        set to zero first."""
        return np.sqrt(np.clip(self.frames, 0, None)).astype(np.float32)

    def summary(self):
        """Return a text summary of the scan."""
        extent = self.probe_positions.max(axis=0) - self.probe_positions.min(axis=0)
        lines = [
            'Scan{}'.format(' "{}"'.format(self.name) if self.name else ''),
            '  frames:               {} of {} x {} pixels'.format(self.num_frames, self.frame_size, self.frame_size),
            '  wavelength:           {:.6g} m ({:.6g} keV)'.format(self.wavelength, HC_KEV_M / self.wavelength),
            '  detector distance:    {:.6g} m'.format(self.det_distance),
            '  detector pixel pitch: {:.6g} m'.format(self.det_pixel_pitch),
            '  scan extent:          {:.6g} m by {:.6g} m (rows by columns)'.format(extent[0], extent[1]),
            '  largest count:        {:.6g}'.format(float(self.frames.max())),
        ]
        return '\n'.join(lines)

    def save(self, path):
        """Write the scan to the group ``/scan`` of an HDF5 file.  A file
        that exists keeps its other groups."""
        with h5py.File(path, 'a') as f:
            g = open_group(f, 'scan')
            g.create_dataset('frames', data=self.frames, chunks=(1,) + self.frames.shape[1:], compression='gzip',
                             compression_opts=1, shuffle=True)
            g.create_dataset('probe_positions', data=self.probe_positions)
            g['probe_positions'].attrs['units'] = 'm'
            g['probe_positions'].attrs['order'] = 'row, column'
            g.attrs['wavelength'] = self.wavelength
            g.attrs['det_distance'] = self.det_distance
            g.attrs['det_pixel_pitch'] = self.det_pixel_pitch
            if self.name:
                g.attrs['name'] = self.name

    @classmethod
    def load(cls, path):
        """Read the scan in the group ``/scan`` of an HDF5 file."""
        with h5py.File(path, 'r') as f:
            if 'scan' not in f:
                raise ValueError('{} holds no scan; its groups are: {}'.format(path, ', '.join(f) or 'none'))
            g = f['scan']
            return cls(g['frames'][...], g['probe_positions'][...],
                       wavelength=float(g.attrs['wavelength']),
                       det_distance=float(g.attrs['det_distance']),
                       det_pixel_pitch=float(g.attrs['det_pixel_pitch']),
                       name=g.attrs.get('name'))
