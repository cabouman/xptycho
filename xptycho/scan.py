"""The Scan: the measurement."""
import os

import h5py
import numpy as np

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
        detector_pixel (float): meters, the detector pixel pitch after any
            binning.
        name (str, optional): a label for summaries.
    """

    def __init__(self, frames, positions, *, wavelength=None, energy=None,
                 detector_distance, detector_pixel, name=None):
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
        self.detector_pixel = float(detector_pixel)
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
            '  detector pixel:    {:.6g} m'.format(self.detector_pixel),
            '  scan extent:       {:.6g} m by {:.6g} m (rows by columns)'.format(extent[0], extent[1]),
            '  largest count:     {:.6g}'.format(float(self.frames.max())),
        ]
        return '\n'.join(lines)

    def show(self, directory=None):
        """Plot one frame on a log scale and the map of scan positions.

        Args:
            directory (str, optional): where the figure is saved as
                ``scan.png``.  None shows it without saving.
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        middle = self.num_frames // 2
        im = axes[0].imshow(np.log10(np.clip(self.frames[middle], 0, None) + 1), cmap='viridis')
        axes[0].set_title('frame {}, log10(counts + 1)'.format(middle))
        fig.colorbar(im, ax=axes[0])
        axes[1].plot(self.positions[:, 1] * 1e6, self.positions[:, 0] * 1e6, '.', markersize=3)
        axes[1].invert_yaxis()
        axes[1].set_aspect('equal')
        axes[1].set_xlabel('column (um)')
        axes[1].set_ylabel('row (um)')
        axes[1].set_title('{} scan positions'.format(self.num_frames))
        fig.tight_layout()
        if directory is None:
            plt.show()
        else:
            os.makedirs(directory, exist_ok=True)
            fig.savefig(os.path.join(directory, 'scan.png'), dpi=150)
            plt.close(fig)

    def save(self, path):
        """Write the scan to an HDF5 file that :meth:`load` reads."""
        with h5py.File(path, 'w') as f:
            f.create_dataset('frames', data=self.frames, chunks=(1,) + self.frames.shape[1:])
            f.create_dataset('positions', data=self.positions)
            f['positions'].attrs['units'] = 'm'
            f['positions'].attrs['order'] = 'row, column'
            f.attrs['wavelength'] = self.wavelength
            f.attrs['detector_distance'] = self.detector_distance
            f.attrs['detector_pixel'] = self.detector_pixel
            if self.name:
                f.attrs['name'] = self.name

    @classmethod
    def load(cls, path):
        """Read a scan file written by :meth:`save`."""
        with h5py.File(path, 'r') as f:
            return cls(f['frames'][...], f['positions'][...],
                       wavelength=float(f.attrs['wavelength']),
                       detector_distance=float(f.attrs['detector_distance']),
                       detector_pixel=float(f.attrs['detector_pixel']),
                       name=f.attrs.get('name'))
