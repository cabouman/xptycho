.. _UserAPIDocs:

========
User API
========

xptycho has three objects.  A :class:`~xptycho.Scan` holds the
measurement.  A :class:`~xptycho.PtychoModel` holds the parameters and
the forward model.  Its :meth:`~xptycho.PtychoModel.recon` method
returns a :class:`~xptycho.Sample`: the object and the probe, with the
record of the run.  A ground truth is also a :class:`~xptycho.Sample`.  The design is described on the
`design pages <https://cabouman.github.io/xptycho/>`_.

.. code-block:: python

    scan = xpt.Scan.load('scan.h5')
    model = xpt.PtychoModel.from_scan(scan, probe_modes=2)
    model.set_params(object_data_fit=0.5, mode_schedule=[20], initial_probe_distance=0.3e-6)
    recon = model.recon(scan, iterations=200)        # probe not given, so estimated

    print(recon.summary())
    xpt.view_sample(recon)
    recon.save(OUTPUT_DIR + '/recon.h5')

The model
---------

.. autoclass:: xptycho.PtychoModel
   :members:

The scan
--------

.. autoclass:: xptycho.Scan
   :members:

The sample
----------

.. autoclass:: xptycho.Sample
   :members:

.. autoclass:: xptycho.RunRecord

.. autofunction:: xptycho.nrmse

.. autofunction:: xptycho.match_scale

File format
-----------

A scan and a sample are groups of an HDF5 file.  One file may hold either or both; each
class reads and writes only its own groups and keeps the others.  Lengths are in meters.

.. code-block:: text

    file.h5                 attribute: format_version
        /scan               attributes: wavelength, detector_distance, detector_pitch, name
            frames          (J, n, n) intensities
            positions       (J, 2), row and column of each probe center
        /sample             attributes: pixel_pitch, origin, name
            object          complex64 (rows, cols)
            probe           complex64 (K, n, n)
        /run                attribute: iterations.  Present for a reconstruction.
            parameters      (N, 4) text: name, value, units, origin
            data_error      (iterations,)
            positions       (J, 2), the positions the run used
            coverage        (rows, cols), the accumulated probe weight

Viewing
-------

The viewing functions are separate from the objects they show.  Each makes
figures, puts them on the screen, and returns them; the windows zoom and pan
with the mouse.  The part of the object shown is an argument, chosen in the
script.

.. autofunction:: xptycho.view_scan

.. autofunction:: xptycho.view_sample

.. autofunction:: xptycho.save_figures

Demo tools
----------

.. autofunction:: xptycho.download

.. autofunction:: xptycho.scan_positions

Operators
---------

The building blocks of the forward model.

.. automodule:: xptycho.operators
   :members:
