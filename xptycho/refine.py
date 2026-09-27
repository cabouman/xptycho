"""Scan position refinement."""


def refine_positions(scan, recon, *, max_shift=1.0, step=1.0):
    """Estimate corrected scan positions from a reconstruction.

    For each position, the shift within ``max_shift`` that best matches
    the measured amplitude, given the reconstruction's object and probe,
    is found by a grid search with spacing ``step``.  The scan is not
    modified; the corrected positions are returned, to be given to a
    new scan or a new reconstruction.

    Version 1 searches integer shifts only.  Sub-pixel positions come in
    version 2.

    Args:
        scan (Scan): the measurement.  Not modified.
        recon (Reconstruction): the current object and probe.
        max_shift (float, optional): pixels, the largest shift searched
            along each axis.  Defaults to 1.
        step (float, optional): pixels, the search spacing.  Defaults to 1.

    Returns:
        ndarray: ``(num_frames, 2)`` positions in the scan's units.
    """
    raise NotImplementedError
