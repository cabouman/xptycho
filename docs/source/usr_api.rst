.. _UserAPIDocs:

========
User API
========

xptycho has three objects.  A :class:`~xptycho.Scan` holds the
measurement.  A :class:`~xptycho.PtychoModel` holds the parameters and
the forward model.  Its :meth:`~xptycho.PtychoModel.recon` method
returns a :class:`~xptycho.Reconstruction`: the object, the probe, and
the record of the run.  The design is described on the
`design pages <https://cabouman.github.io/xptycho/>`_.

.. code-block:: python

    scan = xptycho.Scan.load('scan.h5')
    model = xptycho.PtychoModel.from_scan(scan, probe_modes=2)
    model.set_params(object_data_fit=0.5, mode_schedule=[20], initial_probe_distance=0.3e-6)
    recon = model.recon(scan, iterations=200)        # probe not given, so estimated

    print(recon.summary())
    recon.show(OUTPUT_DIR)
    recon.save(OUTPUT_DIR)

The model
---------

.. autoclass:: xptycho.PtychoModel
   :members:

The scan
--------

.. autoclass:: xptycho.Scan
   :members:

The reconstruction
------------------

.. autoclass:: xptycho.Reconstruction
   :members:

.. autofunction:: xptycho.nrmse

.. autofunction:: xptycho.match_scale

Operators
---------

The building blocks of the forward model.

.. automodule:: xptycho.operators
   :members:
