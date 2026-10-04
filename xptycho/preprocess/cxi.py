"""Reader for raw ptychography files in the CXI format (HDF5), the format
of the Coherent X-ray Imaging Data Bank, cxidb.org."""
import h5py
import numpy as np

from ..scan import HC_KEV_M

JOULES_PER_KEV = 1.602176634e-16
DETECTOR = 'entry_1/instrument_1/detector_1'
SOURCE = 'entry_1/instrument_1/source_1'
TRANSLATION = 'entry_1/sample_1/geometry_1/translation'


def load_raw(path):
    """Read the raw frames and the instrument facts of a CXI file.

    Nothing is corrected: the frames are the detector counts, and the
    translations are as recorded.

    Args:
        path (str): the CXI file.

    Returns:
        dict: with

        - ``frames``: ``(J, rows, cols)``, detector counts.
        - ``dark_frames``: ``(D, rows, cols)``, or None when the file has none.
        - ``translations``: float64 ``(J, 3)``, meters, the recorded
          ``(x, y, z)`` of the sample at each frame.
        - ``wavelength``: meters, from the recorded photon energy.
        - ``det_distance``: meters.
        - ``det_pixel_pitch``: meters.  The pixels must be square.
    """
    with h5py.File(path, 'r') as f:
        detector = f[DETECTOR]
        pitch_x, pitch_y = float(detector['x_pixel_size'][()]), float(detector['y_pixel_size'][()])
        if not np.isclose(pitch_x, pitch_y):
            raise ValueError('the detector pixels are {} m by {} m; xptycho needs square pixels'.format(pitch_x, pitch_y))
        energy_kev = float(f[SOURCE]['energy'][()]) / JOULES_PER_KEV
        return dict(
            frames=detector['data'][...],
            dark_frames=detector['data_dark'][...] if 'data_dark' in detector else None,
            translations=np.asarray(f[TRANSLATION][...], dtype=np.float64),
            wavelength=HC_KEV_M / energy_kev,
            det_distance=float(detector['distance'][()]),
            det_pixel_pitch=pitch_x,
        )
