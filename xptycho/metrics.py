"""Error metrics."""
import numpy as np


def match_scale(image, truth, region=None):
    """Return ``image`` times the one complex number that brings it
    closest to ``truth``.  A ptychographic reconstruction is determined
    only up to a complex constant, so this is applied before comparing.

    Args:
        image (ndarray): complex, the reconstruction.
        truth (ndarray): complex, the same shape.
        region (ndarray, optional): only pixels where this is positive
            are used; the result is zero elsewhere.
    """
    if region is not None:
        mask = np.asarray(region) > 0
        image, truth = image * mask, truth * mask
    scale = np.sum(np.conj(image) * truth) / np.sum(np.abs(image) ** 2)
    return (scale * image).astype(np.complex64)


def nrmse(image, truth, region=None):
    """The normalized root mean square error between a reconstruction and
    its truth, after :func:`match_scale`.

    Args:
        image (ndarray): complex, the reconstruction.
        truth (ndarray): complex, the same shape.
        region (ndarray, optional): only pixels where this is positive
            count, for example a reconstruction's ``scanned_region()``.

    Returns:
        float: ``||c * image - truth|| / ||truth||`` over the counted pixels.
    """
    scaled = match_scale(image, truth, region)
    if region is not None:
        truth = truth * (np.asarray(region) > 0)
    return float(np.sqrt(np.sum(np.abs(scaled - truth) ** 2) / np.sum(np.abs(truth) ** 2)))
