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

Reconstruction parameters
-------------------------

Set with :meth:`~xptycho.PtychoModel.set_params`, read with
:meth:`~xptycho.PtychoModel.get_params`, and listed with their values, units, and origins by
:meth:`~xptycho.PtychoModel.print_params`.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - name
     - default
     - meaning
   * - ``object_data_fit``
     - 0.6
     - How far each object patch moves toward the patch that fits its frame, between 0 and 1
       (:math:`\alpha_1` in the papers).
   * - ``probe_data_fit``
     - 0.6
     - The same for each probe copy (:math:`\alpha_2`).
   * - ``probe_weight_exponent``
     - 1.25
     - The power of the probe magnitude that weights a patch when the patches are averaged
       into the object (:math:`\kappa`), between 1 and 2.
   * - ``relaxation``
     - 0.5
     - The step size of the iteration (:math:`\rho`), between 0 and 1.
   * - ``mode_schedule``
     - none
     - The iterations at which a probe mode is added, until the probe has ``probe_modes``
       modes.
   * - ``mode_energy_fraction``
     - 0.05
     - The share of the probe energy a new mode starts with.
   * - ``orthogonalize_modes``
     - off
     - Replace the modes by an orthogonal set each time a mode is added.
   * - ``initial_probe_distance``
     - none
     - Meters.  The Fresnel propagation distance used in computing the starting probe and a
       new mode.  None means no propagation.
   * - ``object_shape``
     - from the positions
     - The rows and columns of the object grid.  By default the smallest grid that holds
       every patch.
   * - ``object_origin``
     - from the positions
     - Meters.  The position of the center of the first pixel of the object grid.
   * - ``batch_size``
     - from the memory
     - The positions processed together on a device.

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
