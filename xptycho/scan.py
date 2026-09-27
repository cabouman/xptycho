"""The Scan: the measurement, the instrument facts, and the object grid."""

import numpy as np


class Scan:
    """A ptychographic scan: the diffraction frames, the scan positions,
    the instrument facts, and the object grid they imply.

    A ``Scan`` is the input to every reconstruction.  Its frames are
    detector counts, one frame per scan position.  They are held in
    memory when the scan is small and read from a file in batches when
    it is not; the reconstruction sees no difference, because it takes
    the frames only through :meth:`batches`.  Dark subtraction, the
    validity mask, the apodization window, and the square root are
    applied to each batch as it is read, so the stored frames stay raw.

    The object grid, the pixel array the reconstruction is computed on,
    is derived from the positions and the frame size with a margin so
    that every patch lies inside it.  The object pixel size is derived
    from the physics: ``wavelength * detector_distance / (frame_size *
    detector_pixel)``, the far-field relation.  When the geometry is not
    given, positions must be declared in pixels and the pixel size is
    reported as not known.

    A ``Scan`` is never modified by a reconstruction.  Refined positions
    come back in the result, not in the scan.

    Args:
        frames (ndarray or FrameSource): detector counts, shape
            ``(num_frames, height, width)``.  An array holds the frames
            in memory; a :class:`FrameSource` reads them from a file in
            batches.
        positions (ndarray): shape ``(num_frames, 2)``, row then column,
            in metres unless ``position_units='pixel'``.
        wavelength (float, optional): metres.  Give this or ``energy``.
        energy (float, optional): photon energy in keV, converted to the
            wavelength.
        detector_distance (float, optional): metres, sample to detector.
        detector_pixel (float, optional): metres, the detector pixel pitch
            after any binning.
        detector_center (tuple of float, optional): ``(row, col)`` of the
            beam center on the detector, in pixels.  Defaults to the
            frame center.
        dark (ndarray, optional): the dark frame in counts, subtracted
            from every frame as it is read.
        mask (ndarray, optional): ``(height, width)`` of bool, True where
            the detector pixel is valid.  Invalid pixels take no part in
            the data fit.
        apodization (ndarray, optional): ``(height, width)`` weight
            multiplied into every frame's amplitude, as the gold-ball
            experiments of the papers do with a Tukey window.
        position_units (str, optional): ``'m'`` (default) or ``'pixel'``.
            Pixels are allowed only when the geometry is not given.
        margin (int, optional): pixels of object grid added beyond the
            outermost patch on every side.  Defaults to 0.
        name (str, optional): a label for summaries and saved files.

    Example:
        .. code-block:: python

            scan = xptycho.Scan.open('goldballs.h5')
            print(scan.summary())
            scan.show('./output/goldballs')
    """

    def __init__(self, frames, positions, *, wavelength=None, energy=None,
                 detector_distance=None, detector_pixel=None, detector_center=None,
                 dark=None, mask=None, apodization=None, position_units='m',
                 margin=0, name=None):
        raise NotImplementedError

    # ---------------------------------------------------------------- sizes
    @property
    def num_frames(self):
        """The number of scan positions."""
        raise NotImplementedError

    @property
    def frame_shape(self):
        """``(height, width)`` of one frame in detector pixels."""
        raise NotImplementedError

    @property
    def positions(self):
        """``(num_frames, 2)`` float64, row then column, in metres, or in
        pixels when the scan was declared in pixels."""
        raise NotImplementedError

    @property
    def pixel_size(self):
        """The object pixel size in metres, or None when the geometry was
        not given."""
        raise NotImplementedError

    @property
    def object_shape(self):
        """``(rows, cols)`` of the object grid: the smallest grid that
        holds every patch, plus the margin."""
        raise NotImplementedError

    @property
    def origin(self):
        """``(row, col)`` of the object grid's first pixel in metres, in
        the coordinates of the positions."""
        raise NotImplementedError

    @property
    def overlap(self):
        """The mean overlap fraction between neighboring probe positions,
        as the papers define it, computed from the positions and the
        frame size (the probe is taken to fill the frame)."""
        raise NotImplementedError

    # -------------------------------------------------------------- reading
    def batches(self, batch_size, device=None):
        """Yield the frames in batches, corrected and on the device.

        This is the one path from storage into a reconstruction.  Each
        batch is corrected (dark subtracted, clipped at zero, masked,
        apodized, square-rooted to amplitude) as it is read, so the same
        loop serves an in-memory array and a file too large to load.

        Args:
            batch_size (int): frames per batch.  The last batch is shorter.
            device (str or torch.device, optional): where the batch is
                placed.  Defaults to the CPU.

        Yields:
            tuple: ``(indices, amplitude, patch_origins)`` with
            ``indices`` the frame indices in the scan, ``amplitude`` a
            float32 tensor ``(batch, height, width)``, and
            ``patch_origins`` an int64 tensor ``(batch, 2)`` of the
            integer object-grid row and column where each patch starts.
        """
        raise NotImplementedError

    def select(self, indices=None, box=None):
        """Return a sub-scan: the positions listed in ``indices``, or
        those whose patches lie inside ``box``.

        Args:
            indices (sequence of int, optional): frame indices to keep.
            box (tuple, optional): ``(row_min, row_max, col_min, col_max)``
                in metres (or pixels for a pixel-declared scan).

        Returns:
            Scan: a new scan sharing the frame storage of this one.
        """
        raise NotImplementedError

    # ---------------------------------------------------------- inspection
    def parameters(self):
        """Return the scan's facts as rows with provenance.

        Returns:
            list of dict: one row per fact with keys ``name``, ``value``,
            ``units``, ``origin`` (``given``, ``file``, ``derived``, or
            ``default``), and ``note`` (the formula for a derived value,
            or the file a value was read from).
        """
        raise NotImplementedError

    def summary(self):
        """Return a text summary: frame count and size, pixel size and
        where it came from, scan extent in pixels and micrometres, the
        object grid, the overlap fraction, and where the frames live."""
        raise NotImplementedError

    def show(self, directory=None):
        """Plot one diffraction frame on a log scale, the position map,
        and the coverage of the object grid.

        Args:
            directory (str, optional): where the figures are saved.  None
                shows them without saving.
        """
        raise NotImplementedError

    # ------------------------------------------------------------- storage
    def save(self, path):
        """Write the scan to an HDF5 file, streaming the frames.

        The file holds the raw counts (chunked one frame per chunk), the
        positions with their units, the dark frame, the mask, the
        apodization, the geometry, and the provenance of every value.
        :meth:`open` reads it back.

        Args:
            path (str): the file to write.
        """
        raise NotImplementedError

    @classmethod
    def open(cls, path, **facts):
        """Open a scan file without loading the frames.

        The file may be one written by :meth:`save` or a CXI file; the
        format is detected.  Frames are read in batches on demand.  Any
        instrument fact passed as a keyword overrides the file's value
        and is recorded with origin ``given``.

        Args:
            path (str): an xptycho scan file or a CXI file.
            **facts: any constructor argument, to override the file.

        Returns:
            Scan
        """
        raise NotImplementedError

    @classmethod
    def from_tiff_folder(cls, folder, *, positions_file='Translations.tsv.txt',
                         position_columns=('FCx', 'FCy'), **facts):
        """Read the per-frame TIFF layout of the original ptycho_pmace code.

        The folder holds ``frame_data/`` with one TIFF per frame, sorted
        by the integer in each filename, and a translation table whose
        columns are the column and row of each position in pixels.

        Args:
            folder (str): the folder holding ``frame_data/`` and the table.
            positions_file (str, optional): the table's filename.
            position_columns (tuple of str, optional): the column names,
                given as (column, row); they are swapped to (row, column).
            **facts: any constructor argument, for example the geometry.

        Returns:
            Scan
        """
        raise NotImplementedError


class FrameSource:
    """Frames that live in a file and are read in batches.

    A ``FrameSource`` stands in for the frame array of a :class:`Scan`
    whose data does not fit in memory.  It knows the file, the frame
    count and shape, and how to read a contiguous range of frames.  The
    reconstruction never sees it directly; it sees :meth:`Scan.batches`.

    Args:
        path (str): the file.
        dataset (str): the dataset inside the file holding the frames.
    """

    def __init__(self, path, dataset):
        raise NotImplementedError

    def read(self, start, stop):
        """Return frames ``start`` to ``stop`` as a numpy array of counts."""
        raise NotImplementedError
