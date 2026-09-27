"""Ground truth and simulated scans."""


class GroundTruth:
    """A known object and probe with their pixel size, for simulation and
    for scoring a reconstruction.

    Args:
        object (ndarray): complex ``(rows, cols)`` transmittance.
        probe (ndarray): complex ``(height, width)`` or
            ``(modes, height, width)``.
        pixel_size (float): metres, the object pixel of both arrays.
        name (str, optional): a label.
    """

    def __init__(self, object, probe, pixel_size, name=None):
        raise NotImplementedError


def scan_positions(grid, step, max_offset=0.0, seed=0):
    """Return the positions of a rectangular scan with random offsets.

    Args:
        grid (tuple of int): positions along each axis.
        step (float): metres between positions.
        max_offset (float, optional): metres, the largest uniform random
            offset along each axis.  Defaults to 0.
        seed (int, optional): the random seed.

    Returns:
        ndarray: ``(grid[0] * grid[1], 2)`` in metres, row then column,
        centered on zero.
    """
    raise NotImplementedError


def simulate_scan(truth, positions=None, *, grid=None, step=None, max_offset=0.0,
                  wavelength=None, energy=None, peak_photons=None, dark_rate=0.0, seed=0):
    """Simulate a ptychographic scan of a known object.

    A :class:`~xptycho.FarFieldModel` is built from the truth's pixel
    size and probe, its :meth:`~xptycho.PtychographyModel.forward_project`
    gives the exact amplitudes, and Poisson counts are drawn from the
    intensities scaled so the brightest sample is ``peak_photons``, plus
    a Poisson dark term of mean ``dark_rate`` per pixel.  Without
    ``peak_photons`` the frames are the exact intensities.  Counts are
    drawn on the CPU with numpy, so a seed gives the same frames on
    every machine.

    Args:
        truth (GroundTruth): the object and probe.
        positions (ndarray, optional): ``(num_positions, 2)`` in metres.
            Give this, or ``grid`` and ``step``.
        grid (tuple of int, optional): as in :func:`scan_positions`.
        step (float, optional): metres between positions.
        max_offset (float, optional): metres.
        wavelength (float, optional): metres.  Give this or ``energy``.
        energy (float, optional): keV.  The detector distance and pixel
            are then chosen so that the object pixel of the simulation
            equals ``truth.pixel_size``, and recorded as ``derived``.
        peak_photons (float, optional): photons at the brightest sample.
        dark_rate (float, optional): mean dark counts per pixel.
        seed (int, optional): the random seed for the offsets and the
            counts.

    Returns:
        Scan: with the frames in memory and the geometry recorded.
    """
    raise NotImplementedError
