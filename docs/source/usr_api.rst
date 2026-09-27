.. _UserAPIDocs:

========
User API
========

A reconstruction is three steps, and the API has one part for each:

1. **Load or simulate** the scan into a :class:`~xptycho.Scan`
   (:ref:`ScanDocs`).  Measured data goes through
   :func:`~xptycho.preprocess` first.
2. **Reconstruct** with :func:`~xptycho.pmace` (:ref:`PmaceDocs`).
3. **Review** the returned :class:`~xptycho.Reconstruction`
   (:ref:`ReconstructionDocs`).

In outline, every reconstruction script looks like this:

.. code-block:: python

    # 1. Load, or simulate.
    scan = xptycho.Scan.open('scan.h5')

    # 2. Reconstruct.  The probe is known and held fixed, or estimated.
    recon = xptycho.pmace(scan, probe=known_probe, iterations=100)
    recon = xptycho.pmace(scan, fit_probe=True, probe_modes=2,
                          add_mode_at=20, iterations=200)

    # 3. Review.
    print(recon.summary())
    recon.show(OUTPUT_DIR)
    recon.save(OUTPUT_DIR)

Streaming and large scans
-------------------------

The script above does not change when the scan is too large for
memory.  A :class:`~xptycho.Scan` opened from a file keeps its frames
in the file and reads them in batches through
:meth:`~xptycho.Scan.batches`, the only path from storage into a
reconstruction.  The PMACE iteration accumulates its consensus sums
across the batches and divides once per iteration, so the result does
not depend on the batch size.  The per-position state of the
iteration, which is larger than the frames, is kept on the GPU, in
host memory, or in a file in the output folder, whichever fits, and
the choice is printed.  On a node with several GPUs the positions are
split across them.  All of these are machine settings: they never
change the answer, they are chosen automatically, and
:class:`~xptycho.Machine` overrides them.

.. _ScanDocs:

Scan
----

.. autoclass:: xptycho.Scan
   :members:

.. autoclass:: xptycho.FrameSource
   :members:

.. autofunction:: xptycho.preprocess

.. _PmaceDocs:

Reconstruction functions
------------------------

.. autofunction:: xptycho.pmace

.. autoclass:: xptycho.Machine
   :members:

.. autofunction:: xptycho.refine_positions

.. _ReconstructionDocs:

Reconstruction
--------------

.. autoclass:: xptycho.Reconstruction
   :members:

.. autofunction:: xptycho.nrmse

Simulation and ground truth
---------------------------

.. autoclass:: xptycho.GroundTruth

.. autofunction:: xptycho.simulate_scan

.. autofunction:: xptycho.initial_probe

.. autofunction:: xptycho.initial_object

Data
----

.. autofunction:: xptycho.demo_data

.. autofunction:: xptycho.fetch

.. autofunction:: xptycho.download_and_extract

Operators
---------

The building blocks of the algorithms, public so that a new algorithm
can be written from them.

.. automodule:: xptycho.operators
   :members:
