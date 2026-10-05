xptycho: Ptychographic Reconstruction with PMACE
=================================================

**xptycho** is a Python package that reconstructs the complex image of an
object from a ptychographic scan.

**Key features:**

* Reconstructs the magnitude and phase of the object by projected
  multi-agent consensus equilibrium (PMACE).
* Estimates the probe when it is not known, with one or several probe modes.
* Three objects and a few lines of Python: a scan, a model, a reconstruction.
* Runs through PyTorch_ on a CPU, on one GPU, or divided among the GPUs of one node.
* Reproduces the results of the PMACE papers, about 10 to 30 times faster
  than the reference code on the same computer.

.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Simple interface
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      A reconstruction is a few lines: load a scan, build a model, call ``recon``.

   .. grid-item-card:: Known or unknown probe
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Give the probe to hold it fixed, or leave it out and it is estimated with the object.

   .. grid-item-card:: One GPU or several
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      The same script runs on a laptop and on a multi-GPU node, with the same result.

.. grid:: 3

   .. grid-item-card:: :octicon:`rocket;1.5em` Getting Started
      :columns: 12 6 6 4
      :link: overview
      :link-type: doc

      What xptycho does, and your first reconstruction.

   .. grid-item-card:: :octicon:`book;1.5em` User Guide
      :columns: 12 6 6 4
      :link: install
      :link-type: doc

      Installation, the demos, and the interface.

   .. grid-item-card:: :octicon:`tools;1.5em` Developer Docs
      :columns: 12 6 6 4
      :link: dev_design
      :link-type: doc

      How the package was designed, and how it is released.

.. _PyTorch: https://pytorch.org

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Background

   overview
   quick_start
   theory
   credits

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   install
   demos
   usr_api
   usr_preprocess

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Developer Guide

   dev_design
   dev_release
