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
        positions (ndarray): ``(num_frames, 2)``, the row and column of
            the center of each probe position on the object, in meters.
        wavelength (float, optional): meters.  Give this or ``energy``.
        energy (float, optional): keV.
        detector_distance (float): meters, object to detector.
        detector_pitch (float): meters, the detector pixel pitch after any
            binning.
        name (str, optional): a label for summaries.

    In a file, a scan is the group ``/scan`` of an HDF5 file, with datasets
    ``frames`` and ``positions`` and attributes ``wavelength``,
    ``detector_distance``, ``detector_pitch``, and ``name``.  The same file
    may also hold a :class:`~xptycho.Sample` in ``/sample``.
    """

    def __init__(self, frames, positions, *, wavelength=None, energy=None,
                 detector_distance, detector_pitch, name=None):
        frames = np.asarray(frames)
        positions = np.asarray(positions, dtype=np.float64)
        if frames.ndim != 3 or frames.shape[1] != frames.shape[2]:
            raise ValueError('frames must have shape (num_frames, size, size); got {}'.format(frames.shape))
        if positions.shape != (frames.shape[0], 2):
            raise ValueError('positions must have shape ({}, 2); got {}'.format(frames.shape[0], positions.shape))
        if (wavelength is None) == (energy is None):
            raise ValueError('give exactly one of wavelength (meters) and energy (keV)')
        self.frames = frames
        self.positions = positions
        self.wavelength = float(wavelength) if wavelength is not None else energy_to_wavelength(energy)
        self.detector_distance = float(detector_distance)
        self.detector_pitch = float(detector_pitch)
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
        extent = self.positions.max(axis=0) - self.positions.min(axis=0)
        lines = [
            'Scan{}'.format(' "{}"'.format(self.name) if self.name else ''),
            '  frames:            {} of {} x {} pixels'.format(self.num_frames, self.frame_size, self.frame_size),
            '  wavelength:        {:.6g} m ({:.6g} keV)'.format(self.wavelength, HC_KEV_M / self.wavelength),
            '  detector distance: {:.6g} m'.format(self.detector_distance),
            '  detector pitch:    {:.6g} m'.format(self.detector_pitch),
            '  scan extent:       {:.6g} m by {:.6g} m (rows by columns)'.format(extent[0], extent[1]),
            '  largest count:     {:.6g}'.format(float(self.frames.max())),
        ]
        return '\n'.join(lines)

    def show(self, directory=None, block=True):
        """Plot one frame on a log scale and the map of scan positions, with
        a caption, and put the figure on the screen.

        Args:
            directory (str, optional): where the figure is also saved, as
                ``scan.png``.
            block (bool, optional): wait until the window is closed.  False
                leaves it open and returns.  Defaults to True.
        """
        from .view import show_scan
        show_scan(self, directory, block)

    def save(self, path):
        """Write the scan to the group ``/scan`` of an HDF5 file.  A file
        that exists keeps its other groups."""
        with h5py.File(path, 'a') as f:
            g = open_group(f, 'scan')
            g.create_dataset('frames', data=self.frames, chunks=(1,) + self.frames.shape[1:])
            g.create_dataset('positions', data=self.positions)
            g['positions'].attrs['units'] = 'm'
            g['positions'].attrs['order'] = 'row, column'
            g.attrs['wavelength'] = self.wavelength
            g.attrs['detector_distance'] = self.detector_distance
            g.attrs['detector_pitch'] = self.detector_pitch
            if self.name:
                g.attrs['name'] = self.name

    @classmethod
    def load(cls, path):
        """Read the scan in the group ``/scan`` of an HDF5 file."""
        with h5py.File(path, 'r') as f:
            if 'scan' not in f:
                raise ValueError('{} holds no scan; its groups are: {}'.format(path, ', '.join(f) or 'none'))
            g = f['scan']
            return cls(g['frames'][...], g['positions'][...],
                       wavelength=float(g.attrs['wavelength']),
                       detector_distance=float(g.attrs['detector_distance']),
                       detector_pitch=float(g.attrs['detector_pitch']),
                       name=g.attrs.get('name'))
