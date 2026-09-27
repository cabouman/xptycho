xptycho: Ptychographic Reconstruction with PMACE
=================================================

**Reconstruct the complex image of an object from a ptychographic
scan, on one GPU or many.**

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

   scan = xptycho.Scan.open('scan.h5')          # frames, positions, geometry
   model = xptycho.FarFieldModel.from_scan(scan, estimate_probe=True, probe_modes=2)
   recon = model.recon(scan, method='pmace', iterations=200)
   recon.show()
   image = recon.object                         # complex, with recon.pixel_size

What xptycho gives you
----------------------

- **The complex object image** with its pixel size and origin, ready
  for a 3-D laminography reconstruction in mbirtorch.
- **The probe modes** with their energy fractions.
- **A parameter table with provenance**: which values were given,
  read from the file, derived, defaulted, or estimated.
- **Scale**: frames are read in batches, so a scan larger than memory
  reconstructs on one GPU, and the positions can be split across the
  GPUs of one node.

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
