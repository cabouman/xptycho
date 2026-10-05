# xptycho

xptycho: ptychographic reconstruction with projected multi-agent consensus
equilibrium (PMACE) using [PyTorch](https://pytorch.org/).

Features include:
* Reconstruction of the complex object image, magnitude and phase, from far-field diffraction frames.
* Estimation of the probe together with the object, with one or more probe modes.
* Refinement of the probe positions.
* Preprocessing of raw detector frames: dark subtraction, outlier removal, centering, and cropping.
* One HDF5 file format for scans, samples, and reconstructions.
* Demos on simulated data and on measured data, from raw file to image.
* Seamless operation on 1 or more GPUs, Mac MPS, or CPU.

Full documentation at [https://xptycho.readthedocs.io/](https://xptycho.readthedocs.io/)

Design pages at [https://cabouman.github.io/xptycho/](https://cabouman.github.io/xptycho/)

Install from the repository:
```bash
git clone git@github.com:cabouman/xptycho.git
cd xptycho
pip install .
```

Reconstruct in a few lines:
```python
import xptycho as xpt
scan = xpt.Scan.load('scan.h5')
model = xpt.PtychoModel.from_scan(scan)
recon = model.recon(scan)
xpt.view_sample(recon)
```

xptycho implements the PMACE method of Qiuchen Zhai, Gregery T. Buzzard,
Kevin Mertes, Brendt Wohlberg, and Charles A. Bouman.  For the papers to
cite, the source of the demo data, and the funding support, see
[Credits](https://xptycho.readthedocs.io/en/latest/credits.html).
