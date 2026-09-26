.. _DataDocs:

====
Data
====

The demos download what they need on first run.  Downloads go to the
directory named by the environment variable ``XPTYCHO_DATA_DIR``, or
to ``demo/input`` when it is not set.  A download is checked against
its recorded size and checksum, resumes if interrupted, and never asks
a question.  When there is no network, the error names the file, the
URL, and the place to put it by hand.

Datasets
--------

.. list-table::
   :header-rows: 1
   :widths: 22 12 44 22

   * - name
     - size
     - contents
     - source
   * - ``demo_xptycho_data``
     - about 35 MB
     - ground-truth objects and probes for the simulated demos, the
       gold-ball reference probe, and four preprocessed gold-ball
       reference frames
     - Purdue data depot
   * - ``goldballs_cxi``
     - 603 MB
     - the raw gold-ball scan: 820 frames of 621 x 621 counts, of which
       20 are dark frames, with positions and geometry
     - CXIDB ID 65, public domain (CC0)

The gold-ball scan was measured by Stefano Marchesini at ALS beamline
5.3.2 and is cited as: S. Marchesini, "Ptychography Gold Ball Example
Dataset," CXIDB ID 65, Lawrence Berkeley National Laboratory, 2017,
doi:10.11577/1454414.

Scan files
----------

xptycho stores a scan in one HDF5 file holding the raw counts, the
dark frame, the mask, the positions in metres, the geometry, and the
provenance of each value.  :func:`xptycho.Scan.open` reads that file
and also reads CXI files.  A folder of per-frame TIFFs with a
translation table, the layout of the original ptycho_pmace code, is
read by :func:`xptycho.Scan.from_tiff_folder`.
