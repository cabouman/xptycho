"""Preprocessing of raw frames into a Scan."""


def preprocess(raw, *, dark='mean_of_dark_frames', drop_outliers=0, crop=None,
               center='centroid', apodization=None, apodization_shape=0.5, mask=None,
               **facts):
    """Turn raw detector frames into a :class:`Scan`, with every choice named.

    Measured data needs a few corrections before reconstruction, and
    they change the answer, so they are made here, once, visibly, and
    recorded in the scan's parameter table.  The output is a scan whose
    frames are still counts; the corrections are stored beside them and
    applied as frames are read.

    Args:
        raw (Scan or str): a scan opened from a CXI file, or its path.
            Dark frames in the file are recognized by the file's own
            labels.
        dark (str or ndarray, optional): ``'mean_of_dark_frames'`` (the
            default), ``None`` for no subtraction, or a dark frame.
        drop_outliers (int, optional): drop this many frames whose total
            counts are farthest from the median, by a modified z-score.
            Defaults to 0.
        crop (int, optional): crop every frame to this many pixels on a
            side about the detector center.  None keeps the full frame.
        center (str or tuple, optional): ``'centroid'`` of the mean
            frame (the default), ``'frame'`` for the geometric center, or
            ``(row, col)`` in pixels.
        apodization (str, optional): ``'tukey'`` multiplies each frame's
            amplitude by a radial Tukey window, as the gold-ball
            experiments of the papers do.  None applies none.
        apodization_shape (float, optional): the Tukey shape parameter.
            Defaults to 0.5.
        mask (ndarray, optional): valid-pixel mask, True where valid.
        **facts: instrument facts to override the file's, as in
            :class:`Scan`.

    Returns:
        Scan: with the corrections recorded, ready to save or reconstruct.
    """
    raise NotImplementedError
