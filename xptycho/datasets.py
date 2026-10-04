"""Tools for demos: a file download and simple scan patterns."""
import os
import subprocess
import urllib.error
import urllib.request

import numpy as np


def download(url, directory):
    """Download a file into a directory, unless it is already there.

    Args:
        url (str): the address of the file.
        directory (str): the local directory.  It is created if needed.

    Returns:
        str: the path of the local file, named as in the address.
    """
    path = os.path.join(directory, os.path.basename(url))
    if os.path.isfile(path):
        return path
    os.makedirs(directory, exist_ok=True)
    print('downloading {} to {}'.format(url, path))
    partial = path + '.part'
    try:
        urllib.request.urlretrieve(url, partial)
    except urllib.error.URLError:
        # Some servers do not send their intermediate certificate, which
        # Python cannot verify without it; curl can.
        result = subprocess.run(['curl', '-L', '--fail', '-sS', '-o', partial, url], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError('could not download {}: {}'.format(url, result.stderr.strip()))
    os.replace(partial, path)
    return path


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
