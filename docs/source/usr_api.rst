.. _UserAPIDocs:

========
User API
========

xptycho is organized around one idea.  A :class:`~xptycho.PtychographyModel`
is the forward map from an object image to the amplitudes measured at
the scan positions.  An algorithm such as :class:`~xptycho.PMACE` is a
way to invert that map.  One loop reads the scan positions in batches
and runs the chosen algorithm on the chosen model.  Every physical fact
is a model parameter or a model subclass.  Every numerical choice is
part of an algorithm.

A reconstruction is three steps:

1. **Load or simulate** the measurement into a :class:`~xptycho.Scan`
   (:ref:`ScanDocs`).  Measured data goes through
   :func:`~xptycho.preprocess` first.
2. **Build the model** from the scan's geometry and what is known about
   the probe, and **reconstruct** with its
   :meth:`~xptycho.PtychographyModel.recon` method (:ref:`ModelDocs`).
3. **Review** the returned :class:`~xptycho.Reconstruction`
   (:ref:`ReconstructionDocs`).

.. code-block:: python

    scan = xptycho.Scan.open('scan.h5')

    model = xptycho.FarFieldModel.from_scan(scan, probe=known_probe)
    recon = model.recon(scan, method='pmace', iterations=100)

    model = xptycho.FarFieldModel.from_scan(scan, estimate_probe=True, probe_modes=2)
    recon = model.recon(scan, method='pmace', iterations=200, add_mode_iterations=[20])

    print(recon.summary())
    recon.show(OUTPUT_DIR)
    recon.save(OUTPUT_DIR)

Streaming and large scans
-------------------------

The script above does not change when the scan is too large for
memory.  A :class:`~xptycho.Scan` opened from a file keeps its frames
in the file and reads them in batches through
:meth:`~xptycho.Scan.batches`.  A reconstruction reads the frames only
through that method.  The loop accumulates the algorithm's sums across the
batches and reduces them once per pass, so the result does not depend
on the batch size.  The per-position state of the iteration, which is
larger than the frames, is placed on the GPU if it fits there, else
in host memory if it fits there, else in a file.  The choice is
printed.  On a node with
several GPUs the positions are split across them by
:meth:`~xptycho.PtychographyModel.configure_devices`, and one process
drives them all.  A run given a checkpoint directory writes
checkpoints on an interval and resumes from them when the same script
runs again.  See :ref:`LoopDocs` for the loop and the state.

.. _ModelDocs:

The model
---------

.. autoclass:: xptycho.PtychographyModel
   :members:

.. autoclass:: xptycho.FarFieldModel
   :members:

.. _ScanDocs:

The scan
--------

.. autoclass:: xptycho.Scan
   :members:

.. autoclass:: xptycho.FrameStore
   :members:

.. autofunction:: xptycho.preprocess

.. _ReconstructionDocs:

The reconstruction
------------------

.. autoclass:: xptycho.Reconstruction
   :members:

.. autofunction:: xptycho.nrmse

Algorithms
----------

.. autoclass:: xptycho.PMACE
   :members:

.. autofunction:: xptycho.refine_positions

.. _LoopDocs:

The loop and the state
----------------------

For full control of a run, or to write a new algorithm.

.. autoclass:: xptycho.ReconLoop
   :members:

.. autoclass:: xptycho.StateArray
   :members:

.. autoclass:: xptycho.Batch

Simulation and ground truth
---------------------------

.. autoclass:: xptycho.GroundTruth

.. autofunction:: xptycho.scan_positions

.. autofunction:: xptycho.simulate_scan

Data
----

.. autofunction:: xptycho.demo_data

.. autofunction:: xptycho.fetch

.. autofunction:: xptycho.download_and_extract

Operators
---------

The kernels the model is built from, public so that a new model or
algorithm can be written from them.

.. automodule:: xptycho.operators
   :members:
