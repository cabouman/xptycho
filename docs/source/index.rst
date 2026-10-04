xptycho: Ptychographic Reconstruction with PMACE
=================================================

**Reconstruct the complex image of an object from a ptychographic
scan, on one GPU or many.**

.. warning::

   Under construction.  xptycho is being written.  The interface is
   designed and documented, and the reconstruction code is being ported
   from ptycho_pmace.  Nothing here runs yet.

A ptychographic scan records far-field diffraction patterns while a
focused X-ray probe steps across a thin object in overlapping
positions.  xptycho reconstructs the object's complex transmittance,
magnitude and phase, from those patterns, and estimates the probe when
it is not known.  The method is projected multi-agent consensus
equilibrium (PMACE), which treats every scan position as an agent and
finds the image on which all agents agree.

.. plot:: figs/ptycho_geometry.py
   :align: center
   :width: 85%

   A ptychographic scan.  The probe illuminates overlapping patches of
   the object, and the detector records the far-field intensity of
   each patch.

A reconstruction in one screen
------------------------------

.. code-block:: python

   import xptycho

   scan = xptycho.Scan.load('scan.h5')          # frames, positions, instrument facts
   model = xptycho.PtychoModel.from_scan(scan, probe_modes=2)
   model.set_params(mode_schedule=[20], initial_probe_distance=2e-6)
   recon = model.recon(scan, iterations=200)    # no probe given, so it is estimated
   recon.show()
   image = recon.object                         # complex, with recon.pixel_size

What xptycho gives you
----------------------

- **The complex object image** with its pixel size and origin, ready
  for a 3-D laminography reconstruction in mbirtorch.
- **The probe modes** with their energy fractions.
- **A parameter table with provenance**: which values were given,
  read from the file, derived, defaulted, or estimated.
- **Scale**: a reconstruction runs on one GPU or divided among the
  GPUs of one node, with the same result.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   overview
   install
   data
   quick_start
   usr_api
   theory
   credits

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Developer Guide

   dev_release
