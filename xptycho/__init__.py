"""xptycho: ptychographic reconstruction with PMACE in PyTorch.

xptycho reconstructs the complex transmittance image of a thin object
from a ptychographic scan: far-field diffraction patterns recorded
while a focused X-ray probe steps across the object in overlapping
positions.  The method is projected multi-agent consensus equilibrium
(PMACE).  Scans are processed in batches, so the data need not fit in
memory, and the arithmetic runs on CPUs or GPUs through PyTorch.
"""

__version__ = '0.0.1'

__all__ = []
