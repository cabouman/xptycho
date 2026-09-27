"""The PMACE algorithm."""


class PMACE:
    """Projected multi-agent consensus equilibrium: one iteration as
    passes over batches of scan positions.

    Every scan position is an agent that adjusts its patch of the
    object toward its own measurement (the data-fitting step, weighted
    by ``object_data_fit``).  The consensus step averages the patches
    into one image, weighting each pixel by the probe magnitude to the
    power ``probe_weight_exponent``.  The Mann iteration with parameter
    ``relaxation`` alternates them until the agents agree.  With an
    estimated probe, each position also holds its own probe estimate,
    updated the same way and averaged into the probe the object update
    uses; that takes a second pass over the frames per iteration.  See
    :ref:`Theory` for the equations and the symbols.

    An instance holds the algorithm's parameters and, during a run,
    its per-position state.  It is passed to
    :meth:`~xptycho.PtychographyModel.recon` as ``algorithm``, or
    built from ``method='pmace'`` and keyword parameters.

    Args:
        object_data_fit (float, optional): the weight of the data-fitting
            step in the object update, alpha_1, in (0, 1].  Defaults to
            0.6.
        probe_data_fit (float, optional): the same weight in the probe
            update, alpha_2.  Defaults to 0.6.
        probe_weight_exponent (float, optional): the exponent on the probe
            magnitude in the consensus average, kappa, in [1, 2].
            Defaults to 1.25.
        relaxation (float, optional): the Mann relaxation parameter rho,
            in (0, 1).  Defaults to 0.5.
        add_mode_iterations (sequence of int, optional): the iterations
            at which a new probe mode is introduced, from the residual
            intensity the current modes do not explain, until the model's
            ``probe_modes`` is reached.  Defaults to none.
        mode_energy_fraction (float, optional): the fraction of the total
            probe energy given to a new mode; all modes are rescaled so
            the total is unchanged.  Defaults to 0.05.
        propagation_distance (float, optional): metres, the Fresnel
            propagation applied to a new mode.  None applies none.
        probe_state_positions (int, optional): how many positions carry a
            per-position probe estimate.  Defaults to all positions up to
            4096, chosen spatially uniformly.  The object update uses only
            the averaged probe, so this bounds memory; it is an
            approximation when it is fewer than all positions.
        denoiser (callable, optional): applied to the consensus object
            each iteration, the regularized variant of the papers'
            unpublished draft.  Takes and returns a complex image.
        seed (int, optional): seed for the probe-state subsample.

    Example:
        .. code-block:: python

            algorithm = xptycho.PMACE(object_data_fit=0.7, probe_weight_exponent=1.5)
            recon = model.recon(scan, algorithm=algorithm, iterations=100)
    """

    #: Passes over the frames per iteration: one with a known probe, two
    #: when the probe is estimated.
    passes_per_iteration = 1

    def __init__(self, *, object_data_fit=0.6, probe_data_fit=0.6,
                 probe_weight_exponent=1.25, relaxation=0.5, add_mode_iterations=(),
                 mode_energy_fraction=0.05, propagation_distance=None,
                 probe_state_positions=None, denoiser=None, seed=0):
        raise NotImplementedError

    def parameters(self):
        """Return the algorithm's parameters as rows with provenance."""
        raise NotImplementedError

    # ---------------------------------------------------- the loop contract
    def start(self, model, scan, init, loop):
        """Prepare a run: declare the per-position state fields, the
        accumulators, and the constants through ``loop``, and set the
        starting object and probe.

        Args:
            model (PtychographyModel): the forward model.
            scan (Scan): the measurement.
            init (ndarray): the starting object.
            loop (ReconLoop): the loop that owns the arrays.
        """
        raise NotImplementedError

    def pass_batch(self, state, batch, pass_index):
        """Do one batch of one pass: read the state rows of the batch,
        accumulate into the pass's sums, and write the state back.
        Nothing computed here is read again inside the same pass.

        Args:
            state (StateArray): this device's per-position state.
            batch (Batch): the batch.
            pass_index (int): which pass of the iteration.
        """
        raise NotImplementedError

    def end_pass(self, state, pass_index):
        """Finish a pass after its sums are reduced across devices:
        divide the sums, update the object or the probe, and return the
        values reported for the iteration."""
        raise NotImplementedError

    def state_dict(self):
        """Return everything a checkpoint must hold beyond the
        per-position state."""
        raise NotImplementedError

    def load_state_dict(self, state):
        """Restore from :meth:`state_dict`."""
        raise NotImplementedError
