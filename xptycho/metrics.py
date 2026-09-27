"""Error metrics."""


def nrmse(image, truth, coverage=None):
    """Return the normalized root mean square error between a
    reconstruction and its ground truth, after removing one global
    complex scale.

    The scale ``c`` minimizing ``||c * image - truth||`` is applied first,
    because a ptychographic reconstruction is determined only up to a
    complex constant.  With ``coverage`` the error is computed over the
    pixels where the coverage is above a fraction of its median, so the
    unlit border does not count.

    Args:
        image (ndarray): complex, the reconstruction.
        truth (ndarray): complex, the same shape.
        coverage (ndarray, optional): the coverage map of the
            reconstruction.  None uses every pixel.

    Returns:
        float: ``||c * image - truth|| / ||truth||`` over the counted
        pixels.
    """
    raise NotImplementedError
