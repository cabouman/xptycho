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
   * - ``summary.txt``, ``parameters.csv``
     - Every parameter of the run with its units and origin, and the outcome.
   * - ``recon.h5``
     - The object, the probe, the positions, and the coverage, with the pixel size.

Demo data
---------

The demos start from known objects and probes, loaded with
:func:`~xptycho.demo_truth`, and simulate their own frames.  The data is one small archive
(about 4 MB) that downloads on first run to ``demo/input``, or to the directory named by the
environment variable ``XPTYCHO_DATA_DIR``.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Name
     - Contents
   * - ``synthetic``
     - The complex object and the probe of the synthetic experiment of the 2023 PMACE paper.
   * - ``blind``
     - The complex object and the two probe modes of the 2025 blind multi-mode paper.

.. code-block:: python

    truth = xptycho.demo_truth('blind')
    truth.object          # complex image
    truth.probe           # complex, (modes, size, size)
    truth.pixel_size      # meters

The papers are cited in :ref:`Credits <CreditsDocs>`.
