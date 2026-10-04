.. _CreditsDocs:

Credits
=======

The PMACE method
----------------

xptycho is a new implementation of the projected multi-agent consensus
equilibrium (PMACE) method for ptychography, developed by Qiuchen Zhai,
Gregery T. Buzzard, Kevin Mertes, Brendt Wohlberg, and Charles A.
Bouman.  It replaces the research code
`ptycho_pmace <https://github.com/cabouman/ptycho_pmace>`_ written by
Qiuchen Zhai.  The method is described in the following papers.

    Qiuchen Zhai, Gregery T. Buzzard, Kevin Mertes, Brendt Wohlberg, and
    Charles A. Bouman, "Projected Multi-Agent Consensus Equilibrium
    (PMACE) with application to ptychography," `IEEE Transactions on
    Computational Imaging, vol. 9, pp. 1058-1070, 2023
    <https://doi.org/10.1109/TCI.2023.3328288>`_.

    Qiuchen Zhai, Gregery T. Buzzard, Kevin Mertes, Brendt Wohlberg, and
    Charles A. Bouman, "Ptychography using blind multi-mode PMACE,"
    `IEEE Transactions on Computational Imaging, vol. 11, pp. 1320-1335,
    2025 <https://doi.org/10.1109/TCI.2025.3609957>`_.

Two conference papers give earlier and shorter accounts.

    Qiuchen Zhai, Gregery T. Buzzard, Kevin Mertes, Brendt Wohlberg, and
    Charles A. Bouman, "Blind multi-mode ptychography using a distributed
    probe estimate," `IEEE International Conference on Image Processing
    (ICIP), pp. 1115-1120, 2025
    <https://doi.org/10.1109/ICIP55913.2025.11084669>`_.

    Qiuchen Zhai, Brendt Wohlberg, Gregery T. Buzzard, and Charles A.
    Bouman, "Projected multi-agent consensus equilibrium for ptychographic
    image reconstruction," `55th Asilomar Conference on Signals, Systems,
    and Computers, pp. 1694-1698, 2021
    <https://doi.org/10.1109/IEEECONF53345.2021.9723357>`_.

Data
----

The measured data of the demos is the gold-ball scan collected by Stefano
Marchesini and colleagues at beamline 5.3.2.1 of the Advanced Light Source,
Lawrence Berkeley National Laboratory, and published in the Coherent X-ray
Imaging Data Bank under the CC0 public domain dedication.  Please cite it
when you use it.

    Stefano Marchesini et al., "Ptychography Gold Ball Example Dataset,"
    `Coherent X-ray Imaging Data Bank, CXIDB ID 65, 2017
    <https://www.cxidb.org/id-65.html>`_,
    `doi:10.11577/1454414 <https://doi.org/10.11577/1454414>`_.

Development
-----------

xptycho is developed by Charles A. Bouman, Brendt Wohlberg, and Gregery T. Buzzard.

Support
-------

The development of this software was supported by:

    * The U.S. Department of Energy through Los Alamos National Laboratory
    * The Showalter Trust
    * The National Science Foundation under Grant CCF-1763896

Citation
--------

Please cite the papers above when referencing the method, and the
following when referencing this software.
::

    @Misc {xptycho,
    author = {Charles A. Bouman and Brendt Wohlberg and Gregery T. Buzzard},
    title = {{xptycho: Ptychographic Reconstruction with PMACE in PyTorch}},
    howpublished = {Software library available from \url{https://github.com/cabouman/xptycho}},
    note = {Version 0.0.1},
    year = 2026
    }
