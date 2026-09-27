"""The reconstruction loop: the pass over batches, the reduction across
devices, the per-position state, the curves, and the checkpoints."""


class Batch:
    """One batch of scan positions on one device.  The loop builds one
    per batch and passes it to the algorithm's ``pass_batch`` method.

    .. list-table::
       :header-rows: 1
       :widths: 18 32 50

       * - attribute
         - type
         - meaning
       * - ``index``
         - int64 ``(batch,)``
         - the frame index in scan order; -1 in a padded slot
       * - ``slot``
         - int64 ``(batch,)``
         - the row of each position in this device's per-position state
       * - ``origin``
         - int64 ``(batch, 2)``
         - the row and column on the object grid where each patch starts
       * - ``amplitude``
         - float32 ``(batch, height, width)``
         - the corrected measured amplitude
       * - ``valid``
         - bool ``(batch,)``
         - False in the padded tail
    """

    def __init__(self, index, slot, origin, amplitude, valid):
        raise NotImplementedError


class StateArray:
    """The per-position state of an iteration, one row per position of
    one device.

    This is the largest array of a reconstruction, eight bytes per
    measured sample for a complex64 field, so it is placed by fit: on
    the device, in pinned host memory, or in a memory-mapped file under
    the checkpoint directory.  The placement is chosen before anything
    is allocated, reported, and recorded in the parameter table.  Rows
    are in storage order, so :attr:`Batch.slot` indexes them directly.

    Args:
        num_rows (int): positions on this device.
        field_shape (tuple of int): the shape of one row, for example
            ``(height, width)``.
        dtype (torch.dtype): the element type.
        device (torch.device): the device the rows are used on.
        placement (str, optional): ``'device'``, ``'host'``, ``'file'``,
            or ``'auto'`` (the default: the first that fits).
        directory (str, optional): the directory of the file when the
            placement is ``'file'``.
    """

    def __init__(self, num_rows, field_shape, dtype, device, placement='auto',
                 directory=None):
        raise NotImplementedError

    @classmethod
    def estimate_bytes(cls, num_rows, field_shape, dtype):
        """Return the bytes one such array needs."""
        raise NotImplementedError

    @property
    def placement(self):
        """Where the rows live: ``'device'``, ``'host'``, or ``'file'``."""
        raise NotImplementedError

    def get(self, slots, out=None):
        """Return the rows ``slots`` on the device."""
        raise NotImplementedError

    def put(self, slots, values):
        """Write ``values`` into the rows ``slots``."""
        raise NotImplementedError

    def save(self, path):
        """Write the rows to one shard file, then rename into place."""
        raise NotImplementedError

    @classmethod
    def load(cls, path, device, placement='auto'):
        """Read a shard file written by :meth:`save`."""
        raise NotImplementedError


class ReconLoop:
    """Runs an algorithm over the scan for a given number of iterations.

    The loop does the work that is the same for every algorithm.  It
    reads the frames in batches, splits the positions across the
    devices, allocates the per-position state arrays, sums the
    algorithm's accumulators across the devices after each pass,
    records the data error and the other curves, writes the log and
    the parameter table, and writes checkpoints.  An algorithm
    implements three methods, ``start``, ``pass_batch``, and
    ``end_pass``.  The loop makes these guarantees to them:

    * Every position is processed once per pass, in storage order, in
      batches of one fixed size.  The last batch is padded and its
      ``valid`` mask marks the padding.
    * The accumulators are set to zero before each pass.  During a pass
      each device adds into its own copy.  After the pass the copies
      are summed across the devices, before ``end_pass`` is called.
    * A value the algorithm computes inside a pass is not readable
      inside that same pass.  Because of this rule the result does not
      depend on the batch size or on the number of devices.  The test
      suite checks this.
    * Checkpoints are written between passes, at a fixed time interval
      and when the process receives SIGTERM.  A resumed run continues
      from the last checkpoint.

    One process runs all the devices, with one worker thread per
    device.  The positions are sorted into a spatially local order and
    each device is given one contiguous range of that order, so each
    device reads one contiguous range of the file and writes into one
    bounded region of the object.

    :meth:`PtychographyModel.recon` builds and runs a loop.  Use this
    class directly to set every option.

    Args:
        model (PtychographyModel): the forward model.
        scan (Scan): the measurement.
        algorithm (object): an algorithm instance such as
            :class:`~xptycho.PMACE`.
        batch_size (int, optional): frames per batch.  Defaults to the
            model's setting, else the largest that fits.
        devices (list, optional): defaults to the model's devices.
        state_placement (str, optional): ``'device'``, ``'host'``,
            ``'file'``, or ``'auto'``.
        report_every (int, optional): iterations between data-error
            reports.  Defaults to 1.
        checkpoint_dir (str, optional): where checkpoints and file-backed
            state go.
        checkpoint_every (float, optional): minutes between checkpoints.
            Defaults to 10.
        resume (str, optional): ``'auto'`` or ``'never'``.
        deterministic (bool, optional): use a deterministic reduction for
            the sums instead of atomic adds.  Slower; for tests.
    """

    def __init__(self, model, scan, algorithm, *, batch_size=None, devices=None,
                 state_placement='auto', report_every=1, checkpoint_dir=None,
                 checkpoint_every=10.0, resume='auto', deterministic=False):
        raise NotImplementedError

    def run(self, iterations, init=None):
        """Run the algorithm and return the :class:`~xptycho.Reconstruction`."""
        raise NotImplementedError

    def state_array(self, name, field_shape, dtype):
        """Allocate one per-position state field on every device, placed
        by fit, and return it (one :class:`StateArray` per device)."""
        raise NotImplementedError

    def accumulator(self, name, shape, dtype):
        """Allocate one sum that is zeroed before each pass and reduced
        across devices after it."""
        raise NotImplementedError

    def constant(self, name, value):
        """Publish a value that is fixed for the duration of a pass and
        updated only in ``end_pass``."""
        raise NotImplementedError

    def summary(self):
        """Return the loop's settings and choices as text: devices, batch
        size, state placement and size, checkpoint policy."""
        raise NotImplementedError
