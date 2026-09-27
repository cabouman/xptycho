"""PMACE reconstruction."""


def pmace(scan, *, probe=None, fit_probe=False, probe_modes=1, add_mode_at=None,
          mode_energy_fraction=0.05, propagation_distance=None, iterations=100,
          object_data_fit=0.6, probe_data_fit=0.6, probe_weight_exponent=1.25,
          relaxation=0.5, init=None, report_every=1, machine=None, seed=0):
    """Reconstruct the object, and the probe when asked, by PMACE.

    Every scan position is an agent that adjusts its patch of the object
    to match its own diffraction pattern.  A consensus step averages the
    patches into one image, weighting each pixel by the probe's
    illumination, and the Mann iteration alternates the two until the
    agents agree.  With ``fit_probe=True`` the probe is estimated in the
    same way, with one probe estimate per scan position averaged into
    the probe the object update uses.  See :ref:`Theory` for the
    equations and the symbols.

    The frames are read in batches through :meth:`Scan.batches`, and the
    consensus sums are accumulated across batches and divided once per
    iteration, so a scan larger than memory reconstructs the same way a
    small one does.  Device, batch size, and where the per-position
    state lives are chosen automatically and printed; see
    :class:`Machine` to override them.

    Args:
        scan (Scan): the measurement.  Not modified.
        probe (ndarray, optional): complex ``(height, width)`` or
            ``(modes, height, width)``.  With ``fit_probe=False`` this is
            the known probe and is held fixed.  With ``fit_probe=True`` it
            is the starting estimate.  When absent and ``fit_probe=True``,
            the probe is initialized from the data: the back-projection
            of the mean amplitude through a constant object, Fresnel
            propagated by ``propagation_distance``.
        fit_probe (bool, optional): estimate the probe together with the
            object (blind ptychography).  Defaults to False.
        probe_modes (int, optional): the number of probe modes at the end
            of the run.  Modes add in intensity and represent partial
            coherence.  Defaults to 1.
        add_mode_at (int, optional): the iteration at which each mode
            beyond the first is introduced, from the residual intensity
            the current modes do not explain.  Required when
            ``probe_modes > 1`` and no multi-mode ``probe`` is given.
        mode_energy_fraction (float, optional): the fraction of the total
            probe energy given to a newly added mode; every mode is
            rescaled so the total is unchanged.  Defaults to 0.05.
        propagation_distance (float, optional): metres, the Fresnel
            propagation applied to an initialized or newly added probe
            mode.  None applies none.
        iterations (int, optional): iterations to run.  This is the only
            stopping rule.  Defaults to 100.
        object_data_fit (float, optional): the weight of the data-fitting
            step in the object update, alpha_1 in the papers, in (0, 1].
            Defaults to 0.6.
        probe_data_fit (float, optional): the same weight in the probe
            update, alpha_2.  Defaults to 0.6.
        probe_weight_exponent (float, optional): the exponent on the probe
            magnitude in the consensus average, kappa, in [1, 2].
            Defaults to 1.25.
        relaxation (float, optional): the Mann relaxation parameter rho,
            in (0, 1).  Defaults to 0.5.
        init (Reconstruction or ndarray, optional): the starting object.
            A ``Reconstruction`` continues that run: its object, probe,
            positions, and curves carry over, and ``iterations`` counts
            from where it stopped.  An array is the starting object image
            alone.  None initializes the object from the data.
        report_every (int, optional): compute the data error and the
            reported object every this many iterations.  Each report
            costs one extra forward transform per mode per position.
            Defaults to 1.
        machine (Machine, optional): overrides for device, batch size,
            state placement, and the probe-state subsample.
        seed (int, optional): seed for the probe-state subsample and any
            other random choice.  Defaults to 0.

    Returns:
        Reconstruction: the object, the probe, the coverage, the
        positions, the curves, and the parameter table.

    Raises:
        ValueError: when ``probe`` is absent and ``fit_probe`` is False;
            when ``probe_modes > 1`` without ``add_mode_at`` or a
            multi-mode probe; when the probe shape does not match the
            frame shape.
        MemoryError: when the per-position state fits neither the device,
            host memory, nor the output folder's disk.  The message names
            the size and the ``Machine`` argument that changes the
            placement.

    Example:
        .. code-block:: python

            recon = xptycho.pmace(scan, probe=known_probe, iterations=100,
                                  object_data_fit=0.7)
            recon = xptycho.pmace(scan, fit_probe=True, probe_modes=2,
                                  add_mode_at=20, iterations=200)
    """
    raise NotImplementedError
