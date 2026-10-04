.. _DemosDocs:

=====
Demos
=====

The demo scripts are in the `demo folder <https://github.com/cabouman/xptycho/blob/main/demo/>`__.
Follow the installation instructions in :ref:`InstallationDocs`, then run any script from the
repository root.  Each is short and self-contained: every parameter is in one block near the
top, so you can change one and rerun to see its effect.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Script
     - What it demonstrates
   * - ``demo_1_simulated_known_probe.py``
     - The basic pipeline with the probe known: build a model, simulate a scan of a known
       object with Poisson noise, reconstruct, and compare with the truth.  The synthetic
       experiment of the 2023 PMACE paper.
   * - ``demo_2_simulated_blind_two_modes.py``
     - Reconstruction from the data alone: the probe is not given, so it is estimated,
       starting from one mode and adding a second at iteration 20.  The two-mode experiment
       of the 2025 paper.

What a demo writes
------------------

Each demo writes one folder, ``demo/output/<script name>``:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - File
     - Contents
   * - ``scan.png``
     - One diffraction frame and the map of scan positions.
   * - ``object.png``
     - The magnitude and phase of the reconstruction above those of the truth, inside the
       scanned region.
   * - ``probe.png``
     - The probe modes, one per row, with the share of the energy in each.
   * - ``data_error.png``
     - The mismatch between the data and the forward model at each iteration.
   * - ``recon.h5``
     - The reconstruction, a :class:`~xptycho.Sample`: the object and the probe, with the
       record of the run (every parameter with its units and origin, the data error at each
       iteration, the positions, and the coverage).

Demo data
---------

The demos start from a known object and probe and simulate their own frames.  Each ground
truth is one HDF5 file holding a :class:`~xptycho.Sample`.  The demo script names the
address and the local folder, downloads the file on first run, and loads it:

.. code-block:: python

    TRUTH_URL = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xptycho_blind.h5'
    DATA_DIR = './demo/input'

    truth = xpt.Sample.load(xpt.download(TRUTH_URL, DATA_DIR))
    truth.object          # complex image
    truth.probe           # complex, (modes, size, size)
    truth.pixel_pitch     # meters

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Contents
   * - ``demo_xptycho_synthetic.h5`` (2 MB)
     - The complex object and the probe of the synthetic experiment of the 2023 PMACE paper.
   * - ``demo_xptycho_blind.h5`` (3 MB)
     - The complex object and the two probe modes of the 2025 blind multi-mode paper.

The papers are cited in :ref:`Credits <CreditsDocs>`.
