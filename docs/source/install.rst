.. _InstallationDocs:

============
Installation
============

Install xptycho from PyPI:

.. code-block:: bash

   pip install xptycho

To get the demos and the tests as well, install from the repository:

.. code-block:: bash

   git clone git@github.com:cabouman/xptycho.git
   cd xptycho
   pip install .

Either way installs the Python dependencies (numpy, scipy, torch, h5py,
tifffile, matplotlib) automatically.  On Linux, install the CPU or
CUDA build of torch first if you want to choose it:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

To verify an installation from the repository, run the test suite:

.. code-block:: bash

   pip install pytest
   pytest tests/

The tests take a few seconds and need no data.  For a complete runnable
example with a known ground truth, run:

.. code-block:: bash

   python demo/demo_1_simulated_known_probe.py

It simulates a scan of a known object, reconstructs it, and compares
the result to the truth.  It downloads a small data file on first run.
