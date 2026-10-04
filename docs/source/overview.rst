.. _Overview:

========
Overview
========

**xptycho** reconstructs the complex image of a thin object from a
ptychographic scan.

- **Image quality:** the PMACE method, with the weighting by probe intensity of the published papers.
- **Blind reconstruction:** the probe is estimated with the object when it is not known, with one or several modes.
- **Ease of use:** three objects, and defaults taken from the papers.
- **Speed:** PyTorch runs the reconstruction on a CPU, on one GPU, or divided among the GPUs of one node.

See :ref:`QuickStart` for a first reconstruction and :ref:`DemosDocs` for complete scripts.

What a ptychographic scan is
----------------------------

A focused X-ray probe steps across a thin object in overlapping positions.
At each position a detector records the far-field diffraction pattern.  The
detector measures intensity only, so the phase of each pattern is lost.  The
overlap supplies the missing information: every point of the object appears
in several patterns, and only one object is consistent with all of them.

.. plot:: figs/ptycho_geometry.py
   :align: center
   :width: 85%

   A ptychographic scan.  The probe illuminates overlapping patches of the
   object, and the detector records the far-field intensity of each patch.

What xptycho does
-----------------

Given the frames and the scan positions, xptycho returns the object's complex
transmittance, magnitude and phase, and the probe.  The figure below is made
when this page is built.  A small object is scanned by simulation, and the
reconstruction is given only the 289 frames: no probe and no starting image.

.. plot:: figs/overview_example.py
   :align: center
   :width: 100%

   Left to right: one diffraction frame; the reconstructed phase; the true
   phase; the magnitude of the estimated probe.

The method is projected multi-agent consensus equilibrium (PMACE).  Each scan
position adjusts its own patch of the object to match its own frame.  A
consensus step averages the patches into one image, weighting each pixel by
the probe intensity there.  The iteration repeats the two until the patches
agree.  An unknown probe is estimated the same way, and a partially coherent
probe is represented by several modes whose intensities add.  The equations
are in :ref:`Theory`.

Three objects
-------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Scan
      :columns: 12 12 4 4

      **The measurement.**  The diffraction frames, the scan positions, and
      the instrument facts: wavelength, detector distance, detector pitch.
      Never modified.

   .. grid-item-card:: PtychoModel
      :columns: 12 12 4 4

      **The parameters and the forward model.**  Built from a scan.  Its
      ``recon`` method reconstructs; its ``simulate`` method makes a scan from
      a known sample.

   .. grid-item-card:: Sample
      :columns: 12 12 4 4

      **The object and its probe.**  A ground truth and a reconstruction are
      both a sample; a reconstruction also carries the record of its run.  It
      shows itself, saves itself, and can start another run.

.. code-block:: python

   import xptycho as xpt

   scan = xpt.Scan.load('scan.h5')          # frames, positions, instrument facts
   model = xpt.PtychoModel.from_scan(scan)  # the forward model
   recon = model.recon(scan, iterations=100)    # no probe given: it is estimated
   recon.show()                                 # object, probe, data-error curve
   recon.save('./output/recon.h5')              # the sample and the record of the run

Speed
-----

Measured on one Apple laptop, the new code on its GPU against the reference
code ``ptycho_pmace`` on its CPU, with the same data, parameters, and number of
iterations.  The two give the same result to within rounding.

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - case
     - reference code
     - xptycho
     - ratio
   * - gold balls, 794 frames of 512 x 512, known probe, 100 iterations
     - 30 min
     - 3 min
     - 10
   * - synthetic, 400 frames of 256 x 256, probe estimated, two modes, 200 iterations
     - 26 min
     - 53 s
     - 29

Scale
-----

All the data of a reconstruction is held in GPU memory.  On a node with
several GPUs the scan positions and the object image are divided among them,
so the largest scan that fits grows with the number of GPUs.  The result is
the same, up to rounding, for any number of GPUs.  Choose the devices with
:meth:`~xptycho.PtychoModel.configure_devices`; without a call, every GPU
present is used.

What to read next
-----------------

- :ref:`QuickStart`: install, run a demo, reconstruct your own data.
- :ref:`DemosDocs`: the demo scripts and what each one shows.
- :ref:`UserAPIDocs`: every class and function.
- :ref:`Theory`: the PMACE algorithm.
