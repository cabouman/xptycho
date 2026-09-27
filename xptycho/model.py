"""The forward model: from an object image to the amplitudes at the scan
positions, and the reconstruction that inverts it."""


class PtychographyModel:
    """The forward model of a ptychographic scan and the entry point for
    reconstruction.

    A model holds every physical fact of the measurement: the scan
    positions used, the probe modes, the object grid (shape, origin,
    pixel size), and the instrument facts the pixel size follows from.
    It provides the projections between the object and the data, and
    :meth:`recon`, which inverts them with a chosen algorithm.  The
    measurement itself, the frames, lives in a :class:`~xptycho.Scan`
    and is passed to the methods that need it, so a model can be built
    and used without data, for example to simulate.

    This base class holds everything that does not depend on how light
    propagates from the object to the detector.  A subclass supplies
    the propagator and the relation that gives the object pixel size;
    :class:`FarFieldModel` is the far-field (Fraunhofer) model of the
    PMACE papers.

    Every parameter is recorded with its provenance: ``given`` by the
    user, read from a ``file``, ``derived`` from other facts, a package
    ``default``, or ``estimated`` by a reconstruction.
    :meth:`print_params` shows the table.

    The positions in the model are the positions used.  A
    :class:`~xptycho.Scan` keeps the positions as recorded, and a
    refinement updates the model, never the scan.  The probe is a model
    parameter: with ``estimate_probe=True`` it is an unknown, and after
    a reconstruction the estimate is stored in the model with origin
    ``estimated``.

    Args:
        frame_shape (tuple of int): ``(height, width)`` of a frame in
            detector pixels.
        positions (ndarray): ``(num_positions, 2)``, row then column, in
            metres unless ``position_units='pixel'``.
        wavelength (float, optional): metres.  Give this or ``energy``.
        energy (float, optional): keV, converted to the wavelength.
        detector_distance (float, optional): metres, sample to detector.
        detector_pixel (float, optional): metres, the detector pixel
            pitch after any binning.
        detector_center (tuple of float, optional): ``(row, col)`` of the
            beam center on the detector, in pixels.  Defaults to the
            frame center.
        probe (ndarray, optional): complex ``(height, width)`` or
            ``(modes, height, width)``.  The known probe, or the starting
            estimate when ``estimate_probe`` is True.
        probe_modes (int, optional): the number of probe modes.  Modes
            add in intensity and represent partial coherence.  Defaults
            to 1, or to the number given in ``probe``.
        estimate_probe (bool, optional): the probe is unknown and is
            estimated with the object.  Defaults to False.
        position_units (str, optional): ``'m'`` (default) or ``'pixel'``.
            Pixels are allowed only when the geometry is not given, and
            then the pixel size is reported as not known.
        margin (int, optional): pixels of object grid added beyond the
            outermost patch on every side.  Defaults to 0.

    Example:
        .. code-block:: python

            model = xptycho.FarFieldModel.from_scan(scan, probe=known_probe)
            model.print_params()
            recon = model.recon(scan, iterations=100)
    """

    def __init__(self, frame_shape, positions, *, wavelength=None, energy=None,
                 detector_distance=None, detector_pixel=None, detector_center=None,
                 probe=None, probe_modes=1, estimate_probe=False,
                 position_units='m', margin=0):
        raise NotImplementedError

    @classmethod
    def from_scan(cls, scan, **params):
        """Build a model from the geometry a scan recorded.

        The frame shape, the positions, and the instrument facts come
        from the scan with origin ``file`` (or ``given``, when the scan
        was built from arrays).  Any parameter passed here overrides the
        scan's value and is recorded as ``given``.

        Args:
            scan (Scan): the measurement.
            **params: any constructor argument.

        Returns:
            PtychographyModel: an instance of the class this is called on.
        """
        raise NotImplementedError

    # ------------------------------------------------------------ parameters
    def set_params(self, **params):
        """Set model parameters by name, with origin ``given``.

        Accepts every constructor argument, plus ``object_shape``,
        ``origin``, ``batch_size``, ``devices``, and ``deterministic``.
        Setting the positions or the probe changes the forward model;
        setting the batch size or the devices changes only where the
        arithmetic runs.
        """
        raise NotImplementedError

    def get_params(self, names):
        """Return one parameter, or a list for a list of names."""
        raise NotImplementedError

    def parameters(self):
        """Return the parameter table.

        Returns:
            list of dict: one row per parameter with keys ``name``,
            ``value``, ``units``, ``origin``, and ``note`` (the formula
            of a derived value, or the file a value was read from).
        """
        raise NotImplementedError

    def print_params(self):
        """Print the parameter table."""
        raise NotImplementedError

    def summary(self):
        """Return a text summary: the object grid, the pixel size and its
        origin, the scan extent, the overlap fraction, the probe modes
        and whether they are estimated, and the compute settings."""
        raise NotImplementedError

    # ---------------------------------------------------------- object grid
    @property
    def object_shape(self):
        """``(rows, cols)`` of the object grid."""
        raise NotImplementedError

    @property
    def origin(self):
        """``(row, col)`` of the grid's first pixel in metres, in the
        coordinates of the positions."""
        raise NotImplementedError

    @property
    def pixel_size(self):
        """The object pixel size in metres, or None when not known."""
        raise NotImplementedError

    @property
    def overlap(self):
        """The mean overlap fraction between neighboring probe positions,
        as the papers define it."""
        raise NotImplementedError

    def auto_set_object_grid(self):
        """Derive the object grid from the positions, the frame shape,
        and the margin: the smallest grid holding every patch.  Called
        by the constructor; call it again after changing the positions."""
        raise NotImplementedError

    def scale_object_shape(self, row_scale=1.0, col_scale=1.0):
        """Enlarge or shrink the object grid about its center."""
        raise NotImplementedError

    # ------------------------------------------------------------- the probe
    def auto_set_probe(self, scan, propagation_distance=None):
        """Set the starting probe from the data.

        The mean over positions of the back-propagated measured
        amplitude through a constant object, then propagated by
        ``propagation_distance`` when one is given, which adds the
        curvature a focused probe has.  The result is recorded with
        origin ``derived``.  :meth:`recon` calls this when the probe is
        to be estimated and none was given.

        Args:
            scan (Scan): the measurement.
            propagation_distance (float, optional): metres.
        """
        raise NotImplementedError

    # ------------------------------------------------------- the projections
    def forward_project(self, object, indices=None):
        """Predict the measured amplitude at scan positions.

        The nonlinear forward map: each patch of ``object`` is multiplied
        by each probe mode, propagated to the detector, and the modes are
        summed in intensity.

        Args:
            object (ndarray or Tensor): complex, the object grid.
            indices (sequence of int, optional): which positions.
                Defaults to all.

        Returns:
            ndarray or Tensor: float32 ``(num_positions, height, width)``,
            the amplitude, in the form the object was given.
        """
        raise NotImplementedError

    def project_fields(self, object, indices=None):
        """Apply the linear part of the forward model: the complex field at
        the detector for each position and each mode.

        Returns:
            Tensor: complex ``(num_positions, modes, height, width)``.
        """
        raise NotImplementedError

    def back_project(self, fields, indices=None):
        """Apply the exact adjoint of :meth:`project_fields`, from detector
        fields onto the object grid.  Overlapping patches sum.

        Returns:
            Tensor: complex ``(rows, cols)``.
        """
        raise NotImplementedError

    def project_to_data(self, patches, amplitude, indices=None):
        """Replace the amplitude of the predicted fields of ``patches`` by
        the measured ``amplitude`` and map back to patch space.

        This is the per-position data-fit step every amplitude-projection
        algorithm needs; PMACE blends it with the current patch.

        Args:
            patches (Tensor): complex ``(batch, height, width)``.
            amplitude (Tensor): float32 ``(batch, height, width)``.
            indices (Tensor): the positions of the batch.

        Returns:
            Tensor: complex ``(batch, height, width)``.
        """
        raise NotImplementedError

    def gather(self, object, indices):
        """Extract the patches of ``object`` at the positions ``indices``."""
        raise NotImplementedError

    def scatter(self, patches, indices, out=None):
        """Add ``patches`` into an object-grid array at the positions
        ``indices``; the adjoint of :meth:`gather`."""
        raise NotImplementedError

    def coverage(self, weight=None, indices=None):
        """Accumulate a probe weight over the positions: the denominator
        of the consensus average.  ``weight`` defaults to the probe
        magnitude to the power :attr:`probe_weight_exponent` of the
        algorithm; pass ones to count patches."""
        raise NotImplementedError

    def propagate(self, fields):
        """Propagate exit fields from the object plane to the detector.
        Supplied by the subclass."""
        raise NotImplementedError

    def propagate_back(self, fields):
        """The adjoint of :meth:`propagate`.  Supplied by the subclass."""
        raise NotImplementedError

    def data_error(self, object, scan):
        """Return the normalized root mean square error between the
        measured amplitude and the prediction for ``object``, over the
        whole scan, computed in batches."""
        raise NotImplementedError

    # ------------------------------------------------------- reconstruction
    def recon(self, scan, method='pmace', iterations=100, init=None, report_every=1,
              checkpoint_dir=None, resume='auto', algorithm=None, **method_params):
        """Reconstruct the object, and the probe when it is estimated.

        The frames are read in batches through :meth:`Scan.batches`, and
        the algorithm's sums are accumulated across batches and reduced
        once per pass, so a scan larger than memory reconstructs the way
        a small one does.  The per-position state of the iteration is
        kept on the device, in host memory, or in a file under
        ``checkpoint_dir``, whichever fits, and the choice is printed.
        The devices and the batch size are those set on the model, or
        chosen automatically.

        Args:
            scan (Scan): the measurement.  Not modified.
            method (str, optional): ``'pmace'``.  Names the algorithm class
                when ``algorithm`` is not given.  Defaults to ``'pmace'``.
            iterations (int, optional): iterations to run.  This is the
                only stopping rule.  Defaults to 100.
            init (Reconstruction or ndarray, optional): the starting
                object.  A previous result continues that run; an array is
                the starting image; None initializes from the data.
            report_every (int, optional): compute the data error every
                this many iterations; each report costs one forward
                projection.  Defaults to 1.
            checkpoint_dir (str, optional): where checkpoints and any
                file-backed state are written.  None writes none.
            resume (str, optional): ``'auto'`` (the default) continues from
                a checkpoint in ``checkpoint_dir`` whose parameters match
                this call and starts fresh otherwise; ``'never'`` always
                starts fresh.  A mismatched checkpoint is refused with the
                difference named.
            algorithm (object, optional): an algorithm instance such as
                :class:`~xptycho.PMACE`, for full control of its
                parameters.  Overrides ``method`` and ``method_params``.
            **method_params: parameters of the named algorithm, for
                example ``object_data_fit=0.7``.

        Returns:
            Reconstruction: the object, the probe, the coverage, the
            positions, the curves, and the parameter table.  When the
            probe was estimated, the model's probe is updated too.

        Raises:
            ValueError: when the probe is neither given nor estimated, or
                the probe shape does not match the frame shape.
            MemoryError: when the per-position state fits nowhere; the
                message names the size and what to change.
        """
        raise NotImplementedError

    def recon_direct(self, scan):
        """Return the data-driven starting object: each patch set to the
        norm of its measurement over the norm of the probe, averaged into
        the grid, and smoothed.  This is what :meth:`recon` starts from
        when no ``init`` is given.

        Returns:
            ndarray: complex64 on the object grid.
        """
        raise NotImplementedError

    # -------------------------------------------------------------- compute
    def configure_devices(self, num_devices=None, devices=None):
        """Choose the devices a reconstruction runs on.

        With several CUDA devices on one node the scan positions are
        split across them, contiguously in a spatially local order, and
        the algorithm's sums are reduced once per pass.  One process
        drives all devices.  Defaults to every CUDA device present, else
        the CPU.

        Args:
            num_devices (int, optional): use this many of the visible
                devices.
            devices (list of str, optional): the devices, for example
                ``['cuda:0', 'cuda:1']``.
        """
        raise NotImplementedError

    def estimate_memory(self, scan):
        """Return the bytes each large array of a reconstruction of
        ``scan`` needs and where it would be placed, before anything is
        allocated."""
        raise NotImplementedError


class FarFieldModel(PtychographyModel):
    """The far-field (Fraunhofer) ptychography model of the PMACE papers.

    Propagation from the object plane to the detector is the orthonormal
    centered two-dimensional Fourier transform, and the object pixel
    size follows from ``wavelength * detector_distance / (frame_size *
    detector_pixel)``, recorded with origin ``derived``.

    See :class:`PtychographyModel` for the arguments and methods.
    """

    def propagate(self, fields):
        """The orthonormal centered FFT over the last two axes."""
        raise NotImplementedError

    def propagate_back(self, fields):
        """The inverse orthonormal centered FFT."""
        raise NotImplementedError
