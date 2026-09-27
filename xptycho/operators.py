"""The operators every ptychography algorithm is built from.

These are public so that a new algorithm can be written from them
without touching :class:`~xptycho.Scan` or
:class:`~xptycho.Reconstruction`.  All take and return torch tensors on
one device.  Complex tensors are complex64.
"""


def gather_patches(image, patch_origins, patch_shape):
    """Extract the patches of ``image`` that start at ``patch_origins``.

    Args:
        image (Tensor): complex ``(rows, cols)``.
        patch_origins (Tensor): int64 ``(batch, 2)``, the row and column
            where each patch starts.  Every patch must lie inside the
            image.
        patch_shape (tuple of int): ``(height, width)``.

    Returns:
        Tensor: complex ``(batch, height, width)``.
    """
    raise NotImplementedError


def scatter_patches(patches, patch_origins, image):
    """Add the patches into ``image`` at ``patch_origins``, in place.

    The adjoint of :func:`gather_patches`.  Overlapping patches sum.

    Args:
        patches (Tensor): complex or real ``(batch, height, width)``.
        patch_origins (Tensor): int64 ``(batch, 2)``.
        image (Tensor): ``(rows, cols)`` of the same dtype, accumulated
            into and returned.

    Returns:
        Tensor: ``image``.
    """
    raise NotImplementedError


def fft2c(field):
    """The orthonormal, centered two-dimensional Fourier transform over
    the last two axes: zero frequency at the array center, norm
    preserved.  This is the transform of the measurement model."""
    raise NotImplementedError


def ifft2c(spectrum):
    """The inverse of :func:`fft2c`."""
    raise NotImplementedError


def stable_inverse(field, tolerance=1e-6):
    """Return ``conj(field) / (|field|^2 + eps)`` with
    ``eps = tolerance * mean(|field|^2)``, the regularized reciprocal the
    data-fitting step divides by.

    Args:
        field (Tensor): complex, any shape.
        tolerance (float, optional): dimensionless.  Defaults to 1e-6.

    Returns:
        Tensor: complex, the same shape.
    """
    raise NotImplementedError


def coverage(probe_weight, patch_origins, image_shape):
    """Accumulate the probe weight over every patch position.

    Returns the denominator of the consensus average, ``sum_j P_j^T w``,
    for one probe weight array ``w`` such as ``|probe|^kappa``.

    Args:
        probe_weight (Tensor): real ``(height, width)``.
        patch_origins (Tensor): int64 ``(num_positions, 2)``.
        image_shape (tuple of int): ``(rows, cols)``.

    Returns:
        Tensor: real ``(rows, cols)``.
    """
    raise NotImplementedError
