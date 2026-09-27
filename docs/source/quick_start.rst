.. _QuickStart:

===========
Quick Start
===========

The script below simulates a scan of a known object, reconstructs it
with the probe held known, and compares the result to the truth.  It
is the first demo, ``demo/demo_1_simulated_known_probe.py``.

.. code-block:: python

    import xptycho

    # ---------------------------- Parameters ----------------------------
    ENERGY = 8.8                    # keV
    SCAN_GRID = (12, 12)            # scan positions along each axis
    SCAN_STEP = 68 * 4.52e-9        # m between positions
    MAX_OFFSET = 5 * 4.52e-9        # m, random offset of each position
    PEAK_PHOTONS = 1e4              # peak photons in one frame
    DARK_RATE = 0.5                 # mean dark counts per pixel
    ITERATIONS = 100
    OBJECT_DATA_FIT = 0.7           # PMACE object weight
    PROBE_WEIGHT_EXPONENT = 1.5     # exponent on |probe| in the consensus average
    RELAXATION = 0.5                # Mann step size
    OUTPUT_DIR = './output/demo_1_simulated_known_probe'
    # --------------------------------------------------------------------

    truth = xptycho.demo_data('synthetic')       # ground truth object and probe

    scan = xptycho.simulate_scan(truth, grid=SCAN_GRID, step=SCAN_STEP,
                                 max_offset=MAX_OFFSET, energy=ENERGY,
                                 peak_photons=PEAK_PHOTONS, dark_rate=DARK_RATE, seed=0)
    print(scan.summary())
    scan.show(OUTPUT_DIR)            # one diffraction frame and the position map

    # The forward model: the scan's geometry and the known probe.
    model = xptycho.FarFieldModel.from_scan(scan, probe=truth.probe)
    model.print_params()             # every value with its units and its origin

    # The reconstruction: invert the forward model by PMACE.
    recon = model.recon(scan, method='pmace', iterations=ITERATIONS,
                        object_data_fit=OBJECT_DATA_FIT,
                        probe_weight_exponent=PROBE_WEIGHT_EXPONENT,
                        relaxation=RELAXATION)
    print(recon.summary())
    recon.show(OUTPUT_DIR, compare_to=truth)     # phase, magnitude, error images
    recon.save(OUTPUT_DIR, compare_to=truth)     # summary, parameters, recon.h5, plots

For measured data the workflow has one more step: a script that
preprocesses the raw frames into a scan file, which the reconstruction
script then opens.  The probe is unknown there, so the model estimates
it, with two modes.

.. code-block:: python

    scan = xptycho.Scan.open('goldballs.h5')
    model = xptycho.FarFieldModel.from_scan(scan, estimate_probe=True, probe_modes=2)
    recon = model.recon(scan, method='pmace', iterations=200,
                        object_data_fit=0.5, probe_data_fit=0.6,
                        probe_weight_exponent=1.25, relaxation=0.5,
                        add_mode_iterations=[20], mode_energy_fraction=0.1)
    probe = model.get_params('probe')            # the estimate, origin 'estimated'

Known and unknown
-----------------

A probe passed to the model is known and held fixed.
``estimate_probe=True`` makes it an unknown, estimated with the object,
starting from the array when one is given and from the data otherwise;
the estimate is stored back in the model with origin ``estimated``.
``probe_modes`` sets how many probe modes the model has, and the
algorithm's ``add_mode_iterations`` says when each is introduced.
The device, the batch size, and where the per-position state is kept
are chosen automatically and printed.
