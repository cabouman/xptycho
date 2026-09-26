# Concept of operations for xptycho

The current version is ~/claude-notes/xptycho/conops.md on Charlie's
Mac.  In short:

## What the user provides

1. The scan: diffraction frames as detector counts and the scan
   positions in metres, from a CXI file, a TIFF folder, or arrays.
2. The instrument facts: energy, detector distance, detector pixel,
   detector center, dark frames, mask.
3. What is known about the probe: an array to hold fixed, or the
   instruction to estimate it, with the number of modes.

## What xptycho does

Preprocesses raw frames into a scan file in one visible step;
reconstructs the object, and the probe when asked, by PMACE in
batches over positions; reports every parameter with its provenance.

## What comes back

The complex object image (magnitude and phase) with pixel size and
origin, the probe modes, the coverage map, the positions, the
convergence curves, and the parameter table.  One call writes the
folder a laminography reconstruction in mbirtorch reads.
