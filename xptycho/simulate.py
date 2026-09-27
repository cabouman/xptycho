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


def simulate_scan(truth, *, grid, step, max_offset=0.0, wavelength=None, energy=None,
                  detector_distance=None, peak_photons=None, dark_rate=0.0, seed=0):
    """Simulate a ptychographic scan of a known object.

    The probe steps across the object on a ``grid`` of positions
    ``step`` apart, each moved by a uniform random offset of at most
    ``max_offset``.  Each patch is multiplied by the probe modes,
    propagated to the far field by an orthonormal FFT, and summed in
    intensity over the modes.  With ``peak_photons`` the intensities are
    scaled so the brightest sample of the scan is that many photons and
    Poisson counts are drawn, plus a Poisson dark term of mean
    ``dark_rate`` per pixel; without it the frames are the exact
    intensities.

    The detector geometry is chosen so that the object pixel of the
    simulation equals ``truth.pixel_size``: the detector distance and
    pixel follow from the wavelength and the frame size by the far-field
    relation, and the resulting values are recorded in the scan's
    parameter table with origin ``derived``.

    Args:
        truth (GroundTruth): the object and probe.
        grid (tuple of int): positions along each axis.
        step (float): metres between positions.
        max_offset (float, optional): metres, the largest random offset
            of a position along each axis.  Defaults to 0.
        wavelength (float, optional): metres.  Give this or ``energy``.
        energy (float, optional): keV.
        detector_distance (float, optional): metres.  Defaults to a value
            consistent with the truth's pixel size.
        peak_photons (float, optional): photons at the brightest sample.
            None gives noiseless frames.
        dark_rate (float, optional): mean dark counts per pixel.
        seed (int, optional): the random seed for the offsets and the
            counts.  Frames are drawn on the CPU with numpy, so a seed
            gives the same frames on every machine.

    Returns:
        Scan: with the frames in memory and the geometry recorded.
    """
    raise NotImplementedError


def initial_probe(scan, propagation_distance=None):
    """Return the automatic starting probe of a blind reconstruction.

    The mean over positions of the inverse transform of the measured
    amplitude divided by a constant object, then Fresnel propagated by
    ``propagation_distance`` when one is given, which adds the curvature
    a focused probe has.

    Args:
        scan (Scan): the measurement.
        propagation_distance (float, optional): metres.

    Returns:
        ndarray: complex64 ``(1, height, width)``.
    """
    raise NotImplementedError


def initial_object(scan, probe):
    """Return the automatic starting object.

    Each patch is set to a constant equal to the norm of its measurement
    over the norm of the probe, the constants are averaged into the
    object grid, and the result is smoothed.

    Args:
        scan (Scan): the measurement.
        probe (ndarray): the probe or its starting estimate.

    Returns:
        ndarray: complex64 on the object grid.
    """
    raise NotImplementedError
