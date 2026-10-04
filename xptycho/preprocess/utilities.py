"""The general preprocessing steps.  Each takes arrays and returns arrays."""
import numpy as np
from scipy.signal.windows import tukey


def subtract_dark(frames, dark_frames):
    """Subtract the mean dark frame from every frame and set negative
    values to zero.

    Args:
        frames (ndarray): raw intensities, ``(J, rows, cols)``.
        dark_frames (ndarray): frames recorded with no beam,
            ``(D, rows, cols)``.

    Returns:
        ndarray: float32 ``(J, rows, cols)``.
    """
    dark = np.asarray(dark_frames, dtype=np.float64).mean(axis=0).astype(np.float32)
    return np.clip(np.asarray(frames, dtype=np.float32) - dark, 0, None)


def find_outlier_frames(frames, threshold=2.0):
    """Mark the frames whose mean intensity is far from that of the others.

    The mean of each frame is computed, and a frame is an outlier when its
    mean is more than ``threshold`` standard deviations from the average
    of the means.

    Args:
        frames (ndarray): intensities, ``(J, rows, cols)``.
        threshold (float, optional): standard deviations.  Defaults to 2.

    Returns:
        ndarray: bool ``(J,)``, True for an outlier.  Remove them with
        ``frames[~outlier]`` and the same for their positions.
    """
    means = np.asarray(frames).mean(axis=(1, 2), dtype=np.float64)
    return np.abs(means - means.mean()) > threshold * means.std()


def diffraction_center(frames):
    """The center of the diffraction patterns: the intensity center of mass
    of the mean frame, one center for the whole scan.

    Args:
        frames (ndarray): intensities, ``(J, rows, cols)``.

    Returns:
        tuple of float: ``(row, col)`` in pixels.
    """
    mean = np.asarray(frames).mean(axis=0, dtype=np.float64)
    total = mean.sum()
    rows, cols = np.arange(mean.shape[0]), np.arange(mean.shape[1])
    return float((mean.sum(axis=1) * rows).sum() / total), float((mean.sum(axis=0) * cols).sum() / total)


def crop_frames(frames, center, size):
    """Cut a ``size`` by ``size`` square out of every frame, with ``center``
    at pixel ``(size // 2, size // 2)``, where the forward model puts zero
    frequency.

    Args:
        frames (ndarray): ``(J, rows, cols)``.
        center (tuple): ``(row, col)`` of the diffraction center; rounded to
            whole pixels.
        size (int): the side of the square kept.

    Returns:
        ndarray: ``(J, size, size)``.
    """
    row, col = (int(round(c)) - size // 2 for c in center)
    if row < 0 or col < 0 or row + size > frames.shape[1] or col + size > frames.shape[2]:
        raise ValueError('a {0} x {0} square centered at {1} does not fit in frames of {2} x {3}'.format(
            size, tuple(int(round(c)) for c in center), frames.shape[1], frames.shape[2]))
    return frames[:, row:row + size, col:col + size]


def tukey_window(size, shape=0.5):
    """A circular Tukey window: 1 in the middle, tapering to 0 at radius
    ``size / 2``, made by rotating a 1-D Tukey window about the center.

    It multiplies amplitudes.  Frames are intensities, so they are
    multiplied by its square.

    Args:
        size (int): the window is ``size`` by ``size``.
        shape (float, optional): the fraction of the 1-D window inside the
            taper; 0 is no taper.  Defaults to 0.5.

    Returns:
        ndarray: float32 ``(size, size)``, between 0 and 1.
    """
    half = tukey(size, shape)[size // 2 - 1:]            # from the middle to the edge
    coords = np.linspace(-size / 2, size / 2, size)
    radius = np.sqrt(coords[:, None] ** 2 + coords[None, :] ** 2).astype(int)
    window = np.zeros((size, size), dtype=np.float32)
    inside = radius <= size / 2
    window[inside] = half[radius[inside]]
    return window
