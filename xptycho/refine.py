"""Scan position refinement."""


def refine_positions(model, scan, object, *, max_shift=1.0, step=1.0):
    """Estimate corrected scan positions.

    For each position, the shift within ``max_shift`` whose forward
    projection of ``object`` best matches the measured amplitude is
    found by a grid search with spacing ``step``, using the model's
    probe.  Neither the model nor the scan is modified; the corrected
    positions are returned for ``model.set_params(positions=...)``.

    Version 1 searches integer shifts only.

    Args:
        model (PtychographyModel): the forward model with its probe.
        scan (Scan): the measurement.
        object (ndarray): the current object estimate.
        max_shift (float, optional): pixels, the largest shift searched
            along each axis.  Defaults to 1.
        step (float, optional): pixels, the search spacing.  Defaults to 1.

    Returns:
        ndarray: ``(num_frames, 2)`` positions in metres.
    """
    raise NotImplementedError
