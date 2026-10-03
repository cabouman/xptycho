"""Error metrics."""
import numpy as np


def match_scale(image, truth, coverage=None):
    """Return ``image`` times the one complex number that brings it
    closest to ``truth``.  A ptychographic reconstruction is determined
    only up to a complex constant, so this is applied before comparing.

    Args:
        image (ndarray): complex, the reconstruction.
        truth (ndarray): complex, the same shape.
        coverage (ndarray, optional): only pixels where this is positive
            are used and returned nonzero.
    """
    if coverage is not None:
        mask = coverage > 0
        image, truth = image * mask, truth * mask
    scale = np.sum(np.conj(image) * truth) / np.sum(np.abs(image) ** 2)
    return (scale * image).astype(np.complex64)


def nrmse(image, truth, coverage=None):
    """The normalized root mean square error between a reconstruction and
    its truth, after :func:`match_scale`.

    Args:
        image (ndarray): complex, the reconstruction.
        truth (ndarray): complex, the same shape.
        coverage (ndarray, optional): the coverage map of the
            reconstruction; only pixels where it is positive count.

    Returns:
        float: ``||c * image - truth|| / ||truth||`` over the counted pixels.
    """
    scaled = match_scale(image, truth, coverage)
    if coverage is not None:
        truth = truth * (coverage > 0)
    return float(np.sqrt(np.sum(np.abs(scaled - truth) ** 2) / np.sum(np.abs(truth) ** 2)))
