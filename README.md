# xptycho

**Under construction.**  xptycho is being written.  The interface is
designed and documented, and the reconstruction code is being ported
from ptycho_pmace.  Nothing here runs yet.

xptycho reconstructs the complex transmittance image of a thin object
from a ptychographic scan: far-field diffraction patterns recorded
while a focused X-ray probe steps across the object in overlapping
positions.  The method is projected multi-agent consensus equilibrium
(PMACE).  The object and, when asked, the probe are estimated
together, with one or more probe modes.  The arithmetic runs through
PyTorch on a CPU, on one GPU, or divided among the GPUs of one node.

The output is the complex object image, magnitude and phase both
retained, with its pixel size and origin, in a form that feeds a 3-D
laminography reconstruction in
[mbirtorch](https://github.com/cabouman/mbirtorch).

xptycho is a new implementation of the PMACE method developed by
Qiuchen Zhai, Gregery T. Buzzard, Kevin Mertes, Brendt Wohlberg, and
Charles A. Bouman.  It replaces the research code
[ptycho_pmace](https://github.com/cabouman/ptycho_pmace) written by
Qiuchen Zhai.

Full documentation: [xptycho.readthedocs.io](https://xptycho.readthedocs.io).

## Install

```bash
git clone git@github.com:cabouman/xptycho.git
cd xptycho
pip install .
```

## Citation

Please cite the PMACE paper when referencing the method.

```bibtex
@article{zhai2023pmace,
  title = {Projected Multi-Agent Consensus Equilibrium ({PMACE}) with application to ptychography},
  author = {Qiuchen Zhai and Gregery T. Buzzard and Kevin Mertes and Brendt Wohlberg and Charles A. Bouman},
  journal = {IEEE Transactions on Computational Imaging},
  volume = {9},
  pages = {1058--1070},
  year = {2023},
  doi = {10.1109/TCI.2023.3328288}
}
```

Please also cite the blind multi-mode paper when the probe is
estimated or more than one probe mode is used.

```bibtex
@article{zhai2025blind,
  title = {Ptychography using blind multi-mode {PMACE}},
  author = {Qiuchen Zhai and Gregery T. Buzzard and Kevin Mertes and Brendt Wohlberg and Charles A. Bouman},
  journal = {IEEE Transactions on Computational Imaging},
  volume = {11},
  pages = {1320--1335},
  year = {2025},
  doi = {10.1109/TCI.2025.3609957}
}
```

Please cite the software itself when referencing this package.

```bibtex
@misc{xptycho,
  title = {xptycho: Ptychographic Reconstruction with {PMACE} in {PyTorch}},
  author = {Charles A. Bouman and Brendt Wohlberg},
  howpublished = {Software library available from \url{https://github.com/cabouman/xptycho}},
  note = {Version 0.0.1},
  year = 2026
}
```

GitHub's "Cite this repository" button on the repository page generates the
paper citation from `CITATION.cff`.
