"""xptycho: ptychographic reconstruction with PMACE in PyTorch.

xptycho reconstructs the complex transmittance image of a thin object
from a ptychographic scan: far-field diffraction patterns recorded
while a focused X-ray probe steps across the object in overlapping
positions.  A :class:`Scan` holds the measurement, a :class:`PtychoModel`
holds the parameters and the forward model, and
:meth:`PtychoModel.recon` returns a :class:`Reconstruction`.
"""

__version__ = '0.0.1'

from .scan import Scan, energy_to_wavelength
from .model import PtychoModel
from .reconstruction import Reconstruction
from .metrics import nrmse, match_scale
from .datasets import Truth, demo_truth, scan_positions
from . import operators

__all__ = ['Scan', 'PtychoModel', 'Reconstruction', 'Truth', 'demo_truth', 'scan_positions', 'nrmse',
           'match_scale', 'energy_to_wavelength', 'operators']
