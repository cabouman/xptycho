"""The building blocks of the forward model and of PMACE: the centered
Fourier transform, cutting patches out of an image, adding patches into
an image, and the stable division.

Every function works on torch tensors and on any device.  A patch is
identified by the row and column of its first pixel in the image.
"""
import torch


def fft2c(x):
    """Far-field propagation: the orthonormal 2D DFT over the last two
    axes, with the zero frequency at the center of the array."""
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return torch.fft.ifftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))


def ifft2c(x):
    """The inverse of :func:`fft2c`."""
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return torch.fft.ifftshift(torch.fft.ifft2(x, norm='ortho'), dim=(-2, -1))


def _patch_index(starts, size):
    """Index arrays that select the patches starting at ``starts``."""
    offsets = torch.arange(size, device=starts.device)
    rows = starts[:, 0, None] + offsets                 # (B, size)
    cols = starts[:, 1, None] + offsets                 # (B, size)
    return rows[:, :, None], cols[:, None, :]


def gather_patches(image, starts, size):
    """Cut patches out of an image.

    Args:
        image (Tensor): ``(rows, cols)``.
        starts (Tensor): int64 ``(B, 2)``, the row and column of the first
            pixel of each patch.
        size (int): the patch is ``size`` by ``size``.

    Returns:
        Tensor: ``(B, size, size)``.
    """
    rows, cols = _patch_index(starts, size)
    return image[rows, cols]


def scatter_patches(patches, starts, out):
    """Add patches into an image, in place.  Overlapping patches sum.

    Args:
        patches (Tensor): ``(B, size, size)``.
        starts (Tensor): int64 ``(B, 2)``, as in :func:`gather_patches`.
        out (Tensor): ``(rows, cols)``, the image the patches are added to.

    Returns:
        Tensor: ``out``.
    """
    size = patches.shape[-1]
    rows, cols = _patch_index(starts, size)
    rows = rows.expand(-1, size, size)
    cols = cols.expand(-1, size, size)
    return out.index_put_((rows, cols), patches, accumulate=True)


def rms_epsilon(sum_of_squares, count):
    """The small constant of the stable division: ``1e-6`` times the root
    mean square of the array being divided by.

    Args:
        sum_of_squares: the sum of ``|a|**2`` over the whole array ``a``.
        count: the number of elements of ``a``.
    """
    return 1e-6 * (sum_of_squares / count) ** 0.5


def stable_divide(numerator, denominator, epsilon):
    """Divide without blowing up where the denominator is small:
    ``numerator * conj(denominator) / (|denominator|**2 + epsilon)``."""
    return numerator * denominator.conj() / (denominator.abs() ** 2 + epsilon)
