.. _QuickStart:

===============
Getting Started
===============

This page takes you from installation to a reconstruction of your own data in four steps.

1. Install
----------

Follow the :ref:`installation instructions <InstallationDocs>`.  A conda environment or a
Python virtual environment is recommended.

2. Run the first demo
---------------------

The demo scripts are in the `demo folder <https://github.com/cabouman/xptycho/blob/main/demo/>`__
of the repository.  The first one simulates a scan of a known object, reconstructs it with
the probe known, and compares the result with the truth.  Run it from the repository root::

    python demo/demo_1_simulated_known_probe.py

It downloads a small data file on first run and takes a few seconds on a GPU.  It prints the
scan, the parameters, one line per iteration, and the error to the truth, and it writes
figures to ``demo/output/demo_1_simulated_known_probe``.  If it runs, the installation is
working.  The other demos are listed in :ref:`DemosDocs`.

3. Reconstruct your own data
----------------------------

Put your data in two numpy arrays and three numbers:

- ``frames``: a 3D array with shape ``(positions, size, size)``.  Each frame is a diffraction
  pattern in detector counts, with the dark level subtracted, cropped to an even ``size``
  with the beam at the center.
- ``probe_positions``: a 2D array with shape ``(positions, 2)``: the row and the column of the
  center of the probe on the object at each frame, **in meters**.
- The photon ``energy`` in keV (or the ``wavelength`` in meters), the ``det_distance``
  in meters from the object to the detector, and the ``det_pixel_pitch`` pitch in meters.

These go into a :class:`~xptycho.Scan`.  A :class:`~xptycho.PtychoModel` is built from the
scan, and its :meth:`~xptycho.PtychoModel.recon` method does the reconstruction::

    import xptycho as xpt

    scan = xpt.Scan(frames, probe_positions, energy=8.8, det_distance=2.0, det_pixel_pitch=75e-6)
    print(scan.summary())
    xpt.view_scan(scan)                           # one frame and the map of positions

    model = xpt.PtychoModel.from_scan(scan)
    model.print_params()                          # every parameter, its units, and its origin
    recon = model.recon(scan, iterations=100)     # no probe given, so it is estimated

    xpt.view_sample(recon)                        # the object, the probe, the data-error curve
    recon.save('./output/my_scan.h5')
    image = recon.object                          # complex array; recon.pixel_pitch is in meters

The object pixel pitch is not something you choose.  It follows from the instrument:
wavelength times detector distance, divided by frame size times detector pitch.  Check it in
the printed parameters; if it is wrong, one of the three numbers is wrong.

The result is a :class:`~xptycho.Sample`: the object and the probe it was seen with.
``recon.object`` is the complex image, ``recon.phase`` and ``recon.magnitude`` are its two
parts, and ``recon.probe`` holds the probe modes.  ``recon.run`` is the record of the run;
``recon.run.data_error`` is the mismatch between the data and the forward model at each
iteration.  ``recon.save`` writes an HDF5 file that
:meth:`Sample.load <xptycho.Sample.load>` reads back.

4. Adjust the reconstruction
----------------------------

**The probe.**  If you know the probe, pass it and it is held fixed::

    recon = model.recon(scan, probe=my_probe, iterations=100)

If you have a good guess, pass it as the starting point and it is refined::

    recon = model.recon(scan, init_probe=my_guess, iterations=100)

**The starting probe.**  With no probe given, the starting probe is computed from the
data.  Setting ``initial_probe_distance`` propagates it that many meters, which gives it
the curvature of a focused beam and usually helps::

    model.set_params(initial_probe_distance=2e-6)

**Several probe modes.**  A partially coherent beam needs more than one mode.  Say how many
when you build the model, and at which iterations each extra mode is added::

    model = xpt.PtychoModel.from_scan(scan, num_probe_modes=2)
    model.set_params(mode_schedule=[20], initial_probe_distance=2e-6)
    recon = model.recon(scan, iterations=200)
    print(recon.mode_energies)                    # the share of the energy in each mode

**The data-fit weights.**  ``object_data_fit`` (default 0.6) sets how strongly each
position pulls its patch toward its own frame.  Lower it for noisy data.
``probe_data_fit`` does the same for the probe::

    model.set_params(object_data_fit=0.5, probe_data_fit=0.6)

**More iterations.**  Continue from a result instead of starting over::

    recon = model.recon(scan, init=recon, iterations=100)

**The scan positions.**  If the recorded positions may be off by a pixel or so, test it.
:meth:`~xptycho.PtychoModel.refine_probe_positions` tries shifted positions and reports how much
each one would improve the fit.  It changes nothing until you accept the result::

    new_positions, gain = model.refine_probe_positions(scan, recon.object, recon.probe)
    model.set_params(probe_positions=new_positions)
    recon = model.recon(scan, init=recon, iterations=100)

**The devices.**  Without a call, every GPU on the node is used.  To choose::

    model.configure_devices(num_devices=1)             # one GPU, for a run that repeats exactly
    model.configure_devices(devices=['cpu'])           # no GPU

The parameters are listed with :meth:`~xptycho.PtychoModel.print_params`, and every class and
function is described in :ref:`UserAPIDocs`.
