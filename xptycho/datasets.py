"""The demo data: known objects and probes to simulate from and score
against, and simple scan patterns."""
import json
import os
import subprocess
import tarfile
import urllib.error
import urllib.request

import numpy as np

DATA_URL = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xptycho_data.tgz'
DATA_FOLDER = 'demo_xptycho_data'


class Truth:
    """A known object and probe.

    Attributes:
        object (ndarray): complex64 ``(rows, cols)``.
        probe (ndarray): complex64 ``(K, n, n)``, the probe modes.
        pixel_size (float): meters, the object pixel both were made at.
        name (str): the name of the dataset.
    """

    def __init__(self, object, probe, pixel_size, name):
        self.object = object
        self.probe = probe
        self.pixel_size = pixel_size
        self.name = name


def data_directory():
    """Where demo data is kept: ``$XPTYCHO_DATA_DIR``, else ``./demo/input``."""
    return os.environ.get('XPTYCHO_DATA_DIR', os.path.join('.', 'demo', 'input'))


def fetch():
    """Return the folder of the demo data, downloading it on first use."""
    folder = os.path.join(data_directory(), DATA_FOLDER)
    if os.path.isdir(folder):
        return folder
    os.makedirs(data_directory(), exist_ok=True)
    archive = os.path.join(data_directory(), os.path.basename(DATA_URL))
    print('downloading {} to {}'.format(DATA_URL, archive))
    try:
        urllib.request.urlretrieve(DATA_URL, archive)
    except urllib.error.URLError:
        # The data server does not send its intermediate certificate, which
        # Python cannot verify without it; curl can.
        result = subprocess.run(['curl', '-L', '--fail', '-sS', '-o', archive, DATA_URL], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError('could not download {}: {}.  Download it by hand and extract it in {}'.format(
                DATA_URL, result.stderr.strip(), data_directory()))
    with tarfile.open(archive) as tar:
        tar.extractall(data_directory(), filter='data')
    os.remove(archive)
    return folder


def demo_truth(name):
    """Load a known object and probe from the demo data.

    Args:
        name (str): ``'synthetic'``, the object and probe of the synthetic
            experiment of the 2023 PMACE paper; or ``'blind'``, the object
            and the two probe modes of the 2025 paper.

    Returns:
        Truth: with ``object``, ``probe``, and ``pixel_size``.
    """
    folder = os.path.join(fetch(), name)
    if not os.path.isdir(folder):
        raise ValueError('"{}" is not a demo dataset; the datasets are: {}'.format(name, ', '.join(sorted(os.listdir(fetch())))))
    facts = json.load(open(os.path.join(folder, 'truth.json')))
    probe = np.load(os.path.join(folder, 'probe.npy')).astype(np.complex64)
    if probe.ndim == 2:
        probe = probe[None]
    return Truth(np.load(os.path.join(folder, 'object.npy')).astype(np.complex64), probe, facts['pixel_size'], name)


def scan_positions(grid, step, max_offset=0.0, seed=0):
    """The positions of a rectangular scan, with a random offset at each.

    Args:
        grid (tuple of int): the number of positions along rows and columns.
        step (float): meters between neighboring positions.
        max_offset (float, optional): meters.  Each position is moved by a
            uniform random amount up to this along each axis.  Defaults to 0.
        seed (int, optional): the random seed of the offsets.

    Returns:
        ndarray: ``(grid[0] * grid[1], 2)``, row then column, in meters,
        centered on zero.
    """
    rows = (np.arange(grid[0]) - (grid[0] - 1) / 2) * step
    cols = (np.arange(grid[1]) - (grid[1] - 1) / 2) * step
    positions = np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1).reshape(-1, 2)
    if max_offset:
        positions = positions + np.random.default_rng(seed).uniform(-max_offset, max_offset, positions.shape)
    return positions
