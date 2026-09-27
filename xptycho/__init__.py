"""xptycho: ptychographic reconstruction with PMACE in PyTorch.

xptycho reconstructs the complex transmittance image of a thin object
from a ptychographic scan: far-field diffraction patterns recorded
while a focused X-ray probe steps across the object in overlapping
positions.  The method is projected multi-agent consensus equilibrium
(PMACE).  Scans are processed in batches, so the data need not fit in
memory, and the arithmetic runs on CPUs or GPUs through PyTorch.
"""

__version__ = '0.0.1'

from .scan import Scan, FrameSource
from .reconstruction import Reconstruction
from .machine import Machine
from .pmace import pmace
from .simulate import GroundTruth, simulate_scan, initial_probe, initial_object
from .data import DATASETS, download_and_extract, fetch, demo_data
from .preprocess import preprocess
from .metrics import nrmse
from .refine import refine_positions
from . import operators

__all__ = [
    'Scan', 'FrameSource', 'Reconstruction', 'Machine', 'pmace',
    'GroundTruth', 'simulate_scan', 'initial_probe', 'initial_object',
    'DATASETS', 'download_and_extract', 'fetch', 'demo_data',
    'preprocess', 'nrmse', 'refine_positions', 'operators',
]
