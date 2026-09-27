"""xptycho: ptychographic reconstruction with PMACE in PyTorch.

xptycho reconstructs the complex transmittance image of a thin object
from a ptychographic scan: far-field diffraction patterns recorded
while a focused X-ray probe steps across the object in overlapping
positions.  A :class:`PtychographyModel` is the forward map from an
object image to the amplitudes at the scan positions; an algorithm
such as :class:`PMACE` is a way to invert that map; and one loop
streams the positions through them, so the data need not fit in
memory and the arithmetic runs on CPUs or GPUs through PyTorch.
"""

__version__ = '0.0.1'

from .model import PtychographyModel, FarFieldModel
from .scan import Scan, FrameStore
from .reconstruction import Reconstruction
from .pmace import PMACE
from .loop import ReconLoop, StateArray, Batch
from .simulate import GroundTruth, scan_positions, simulate_scan
from .preprocess import preprocess
from .datasets import DATASETS, download_and_extract, fetch, demo_data
from .metrics import nrmse
from .refine import refine_positions
from . import operators

__all__ = [
    'PtychographyModel', 'FarFieldModel', 'Scan', 'FrameStore', 'Reconstruction',
    'PMACE', 'ReconLoop', 'StateArray', 'Batch',
    'GroundTruth', 'scan_positions', 'simulate_scan', 'preprocess',
    'DATASETS', 'download_and_extract', 'fetch', 'demo_data',
    'nrmse', 'refine_positions', 'operators',
]
