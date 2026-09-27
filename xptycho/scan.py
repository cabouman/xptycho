"""The Scan: the measurement as the instrument recorded it."""


class Scan:
    """A ptychographic measurement: the diffraction frames, the recorded
    scan positions, the corrections, and the instrument facts.

    A ``Scan`` holds the measured data.  The object grid, the positions
    used, and the probe are parameters of the
    :class:`~xptycho.PtychographyModel`, not of the scan.  Frames are
    detector counts, one per position.  They are held in memory when
    the scan is small and read from a file in batches when it is not.
    A reconstruction reads them only through :meth:`batches`, which
    behaves the same in both cases.  Dark subtraction, the validity
    mask, the apodization window, and the square root are applied to
    each batch as it is read.  The stored frames are the raw counts.

    A ``Scan`` is never modified by a reconstruction.

    Args:
        frames (ndarray or FrameStore): detector counts, shape
            ``(num_frames, height, width)``.  An array holds the frames
            in memory; a :class:`FrameStore` reads them from a file.
        positions (ndarray): ``(num_frames, 2)``, row then column, in
            metres unless ``position_units='pixel'``.
        wavelength (float, optional): metres.  Give this or ``energy``.
        energy (float, optional): keV.
        detector_distance (float, optional): metres.
        detector_pixel (float, optional): metres, after any binning.
        detector_center (tuple of float, optional): ``(row, col)`` of the
            beam center in detector pixels.
        dark (ndarray, optional): the dark frame in counts, subtracted
            from every frame as it is read.
        mask (ndarray, optional): ``(height, width)`` bool, True where
            the detector pixel is valid.
        apodization (ndarray, optional): ``(height, width)`` weight
            multiplied into every frame's amplitude.
        position_units (str, optional): ``'m'`` (default) or ``'pixel'``.
        name (str, optional): a label for summaries and saved files.

    Example:
        .. code-block:: python

            scan = xptycho.Scan.open('goldballs.h5')
            print(scan.summary())
            scan.show('./output/goldballs')
    """

    def __init__(self, frames, positions, *, wavelength=None, energy=None,
                 detector_distance=None, detector_pixel=None, detector_center=None,
                 dark=None, mask=None, apodization=None, position_units='m', name=None):
        raise NotImplementedError

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
        """``(num_frames, 2)`` float64, row then column, as recorded."""
        raise NotImplementedError

    def batches(self, batch_size, device=None):
        """Yield the frames in batches, corrected and on the device.

        A reconstruction reads the frames only through this method.
        Each batch is dark subtracted, clipped at zero, masked, apodized,
        and square-rooted to amplitude as it is read.  When the frames
        are in a file, the next batch is read into pinned memory while
        the current batch is used.

        Args:
            batch_size (int): frames per batch.  The last batch is shorter;
                the reconstruction loop pads it.
            device (str or torch.device, optional): where the batch is
                placed.  Defaults to the CPU.

        Yields:
            tuple: ``(indices, amplitude)`` with ``indices`` an int64
            tensor of frame indices in scan order and ``amplitude`` a
            float32 tensor ``(batch, height, width)``.
        """
        raise NotImplementedError

    def select(self, indices):
        """Return a sub-scan of the listed frames, sharing this scan's
        frame storage."""
        raise NotImplementedError

    def parameters(self):
        """Return the recorded facts as rows with provenance: keys
        ``name``, ``value``, ``units``, ``origin`` (``given`` or
        ``file``), and ``note``."""
        raise NotImplementedError

    def summary(self):
        """Return a text summary: frame count and size, the recorded
        instrument facts, the scan extent, which corrections are set,
        and where the frames live."""
        raise NotImplementedError

    def show(self, directory=None):
        """Plot one diffraction frame on a log scale and the position map.

        Args:
            directory (str, optional): where the figures are saved.  None
                shows them without saving.
        """
        raise NotImplementedError

    def save(self, path):
        """Write the scan to an HDF5 file, streaming the frames.

        The file holds the raw counts chunked one frame per chunk, the
        positions with their units, the dark frame, the mask, the
        apodization, the instrument facts, and the provenance of every
        value.  :meth:`open` reads it back.

        Args:
            path (str): the file to write.
        """
        raise NotImplementedError

    @classmethod
    def open(cls, path, **facts):
        """Open a scan file without loading the frames.

        The file may be one written by :meth:`save` or a CXI file; the
        format is detected.  Any instrument fact passed as a keyword
        overrides the file's value and is recorded as ``given``.

        Args:
            path (str): the file.
            **facts: any constructor argument.

        Returns:
            Scan
        """
        raise NotImplementedError

    @classmethod
    def from_tiff_folder(cls, folder, *, positions_file='Translations.tsv.txt',
                         position_columns=('FCx', 'FCy'), **facts):
        """Read the per-frame TIFF layout of the original ptycho_pmace code.

        The folder holds ``frame_data/`` with one TIFF per frame, sorted
        by the integer in each filename, and a table whose columns are
        the column and row of each position in pixels.

        Args:
            folder (str): the folder.
            positions_file (str, optional): the table's filename.
            position_columns (tuple of str, optional): the column names
                as (column, row); they are swapped to (row, column).
            **facts: any constructor argument.

        Returns:
            Scan
        """
        raise NotImplementedError


class FrameStore:
    """Frames in a file, read by contiguous range.

    A :class:`Scan` whose frames do not fit in memory holds a
    ``FrameStore`` in place of the frame array.  A reader for a new
    file format implements the three members below: the frame count,
    the frame shape, and a read of frames ``start`` to ``stop`` into a
    buffer the caller supplies.

    Args:
        path (str): the file.
        dataset (str, optional): the dataset inside the file holding the
            frames, for HDF5-based formats.
    """

    def __init__(self, path, dataset=None):
        raise NotImplementedError

    @property
    def num_frames(self):
        raise NotImplementedError

    @property
    def frame_shape(self):
        raise NotImplementedError

    def read_into(self, start, stop, out):
        """Read frames ``start`` to ``stop`` into ``out``, a preallocated
        array of counts, and return it."""
        raise NotImplementedError
