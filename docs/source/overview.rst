========
Overview
========

What xptycho does
-----------------

A ptychographic scan steps a focused X-ray probe across a thin object
in overlapping positions and records the far-field diffraction pattern
at each one.  The detector measures intensity only, so the phase of
each pattern is lost.  The overlap between neighboring positions
supplies the missing information: every point of the object appears in
several patterns, and only one object is consistent with all of them.

xptycho reconstructs that object, its complex transmittance, by the
PMACE method.  Each scan position is an agent that adjusts its patch
of the object to match its own measurement.  A consensus step averages
the patches back into one image, weighted by the probe's illumination.
The iteration alternates the two until the agents agree.  When the
probe is unknown, it is estimated in the same way, with one probe
estimate per scan position averaged into a global probe.  A partially
coherent probe is represented by several modes whose intensities add.

What the user provides
----------------------

1. The scan: diffraction frames as detector counts, one per position,
   and the scan positions.  From a CXI file, a folder of TIFF frames
   with a translation table, or arrays in memory.
2. The instrument facts: photon energy, detector distance, detector
   pixel size, and the detector center; dark frames and a bad-pixel
   mask when they exist.  The object pixel size follows from these.
3. What is known about the probe: a probe array to hold fixed, or the
   instruction to estimate it.

What comes back
---------------

The complex object image with its pixel size and origin, the probe
modes, the coverage map, the positions, the convergence curves, and a
parameter table that records where every value came from.

Scale
-----

Frames are read in batches and the consensus sums are accumulated
across batches, so a scan larger than memory reconstructs on one
GPU.  The per-position state of the iteration is kept on the device,
in host memory, or in a file, whichever fits, and the choice is
printed.  On a node with several GPUs the positions are split across
them.

See :ref:`QuickStart` for a complete script and :ref:`Theory` for the
equations.
