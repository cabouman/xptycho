.. _QuickStart:

===========
Quick Start
===========

The script below simulates a scan of a known object, reconstructs it
with the probe known and held fixed, and compares the result with the
truth.  It is the first demo, ``demo/demo_1_simulated_known_probe.py``.

.. literalinclude:: ../../demo/demo_1_simulated_known_probe.py
   :language: python

The second demo, ``demo/demo_2_simulated_blind_two_modes.py``, is given
only the data.  It estimates the object and a probe with two modes.

.. literalinclude:: ../../demo/demo_2_simulated_blind_two_modes.py
   :language: python
   :start-after: # --------------------------------------------------------------------

Known and unknown
-----------------

A probe passed to :meth:`~xptycho.PtychoModel.recon` as ``probe=`` is
known and held fixed.  With no probe given, the probe is estimated with
the object, starting from ``init_probe=`` when one is given and from the
data otherwise.  ``probe_modes`` on the model is the number of modes the
run ends with; an estimate starts with one mode and adds one at each
iteration of the ``mode_schedule`` parameter.  The devices and the batch
size are chosen automatically and printed.
