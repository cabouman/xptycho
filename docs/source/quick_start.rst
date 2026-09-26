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
    DETECTOR_DISTANCE = 2.0         # m
    DETECTOR_PIXEL = 75e-6          # m
    FRAME_SIZE = 256                # detector pixels per side
    SCAN_GRID = (12, 12)            # scan positions
    SCAN_STEP = 1.02e-6             # m between positions
    PEAK_PHOTONS = 1e4              # peak photons in one frame
    ITERATIONS = 100
    OBJECT_DATA_FIT = 0.7           # PMACE object weight
    PROBE_WEIGHT_EXPONENT = 1.25    # exponent on |probe| in the consensus average
    RELAXATION = 0.5                # Mann step size
    OUTPUT_DIR = './output/demo_1_known_probe'
    # --------------------------------------------------------------------

    truth = xptycho.demo_data('synthetic')       # ground truth object and probe

    scan = xptycho.simulate_scan(truth, grid=SCAN_GRID, step=SCAN_STEP,
                                 energy=ENERGY, detector_distance=DETECTOR_DISTANCE,
                                 detector_pixel=DETECTOR_PIXEL, frame_size=FRAME_SIZE,
                                 peak_photons=PEAK_PHOTONS, seed=0)
    print(scan.summary())            # sizes, object grid, pixel size, overlap
    scan.show(OUTPUT_DIR)            # one diffraction frame and the position map

    # The reconstruction: PMACE with the probe held known.
    recon = xptycho.pmace(scan, probe=truth.probe, iterations=ITERATIONS,
                          object_data_fit=OBJECT_DATA_FIT,
                          probe_weight_exponent=PROBE_WEIGHT_EXPONENT,
                          relaxation=RELAXATION)
    print(recon.summary())                       # final errors, time, device chosen
    recon.show(OUTPUT_DIR, compare_to=truth)     # phase, magnitude, error images
    recon.save(OUTPUT_DIR, compare_to=truth)     # summary, parameters, recon.h5, plots

For measured data the workflow has one more step: a script that
preprocesses the raw frames into a scan file, which the reconstruction
script then opens.

.. code-block:: python

    scan = xptycho.Scan.open('goldballs.h5')
    recon = xptycho.pmace(scan, fit_probe=True, probe_modes=2, add_mode_at=20,
                          mode_energy_fraction=0.1, iterations=200,
                          object_data_fit=0.5, probe_data_fit=0.6,
                          probe_weight_exponent=1.25, relaxation=0.5)

Known and unknown
-----------------

A probe passed as a plain array is known and held fixed.
``fit_probe=True`` estimates the probe, starting from the array when
one is given and from an automatic initialization otherwise.
``probe_modes`` sets how many probe modes are estimated, and
``add_mode_at`` the iteration at which the second mode is introduced.
Every other choice, the device, the batch size, and where the
per-position state is kept, is made automatically and printed.
