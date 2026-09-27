"""The stateless tensor kernels the forward model is built from.

These are public so that a new model or algorithm can be written from
them and so that they can be tested alone.  All take and return torch
tensors on one device.  Complex tensors are complex64.
"""


def gather_patches(image, origins, patch_shape):
    """Extract the patches of ``image`` that start at ``origins``.

    Args:
        image (Tensor): complex ``(rows, cols)``.
        origins (Tensor): int64 ``(batch, 2)``, the row and column where
            each patch starts.  Every patch must lie inside the image.
        patch_shape (tuple of int): ``(height, width)``.

    Returns:
        Tensor: complex ``(batch, height, width)``.
    """
    raise NotImplementedError


def scatter_patches(patches, origins, image):
    """Add the patches into ``image`` at ``origins``, in place; the
    adjoint of :func:`gather_patches`.  Overlapping patches sum.

    Args:
        patches (Tensor): ``(batch, height, width)``.
        origins (Tensor): int64 ``(batch, 2)``.
        image (Tensor): ``(rows, cols)`` of the same dtype, accumulated
            into and returned.

    Returns:
        Tensor: ``image``.
    """
    raise NotImplementedError


def fft2c(field):
    """The orthonormal, centered two-dimensional Fourier transform over
    the last two axes: zero frequency at the array center, norm
    preserved.  This is the reference transform of the far-field model;
    the reconstruction loop uses an equivalent form that folds the
    centering into precomputed arrays."""
    raise NotImplementedError


def ifft2c(spectrum):
    """The inverse of :func:`fft2c`."""
    raise NotImplementedError
