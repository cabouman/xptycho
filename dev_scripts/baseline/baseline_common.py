"""Shared pieces of the baseline capture scripts.

The scripts run the OLD ptycho_pmace code in the pmace_repro environment
and write reference outputs that xptycho is checked against.  Each run
writes one self-contained folder: inputs/, iterates/, curves/,
params.txt, summary.txt.
"""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np


def import_pmace():
    """Import the old package, with the bm4d stub if bm4d is absent."""
    try:
        import bm4d  # noqa: F401
    except Exception:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bm4d_stub'))
    import pmace.pmace as pm
    import pmace.utils as pu
    import pmace.nrmse as pn
    return pm, pu, pn


def git_commit(repo_dir):
    """Return the checked-out commit of a repository, or 'unknown'."""
    try:
        return subprocess.check_output(['git', '-C', repo_dir, 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:
        return 'unknown'


def versions():
    import numpy, scipy, pyfftw
    return {'python': platform.python_version(), 'numpy': numpy.__version__,
            'scipy': scipy.__version__, 'pyfftw': pyfftw.__version__,
            'platform': platform.platform()}


def make_run_dir(root, name):
    """Create root/name with the four subfolders and return its path."""
    run_dir = os.path.join(root, name)
    for sub in ('inputs', 'iterates', 'curves', 'old_code_output'):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def save_inputs(run_dir, **arrays):
    for key, value in arrays.items():
        if value is not None:
            np.save(os.path.join(run_dir, 'inputs', key + '.npy'), value)


def write_params(run_dir, params, pmace_dir, seconds):
    params = dict(params)
    params['ptycho_pmace_commit'] = git_commit(pmace_dir)
    params.update(versions())
    params['wall_seconds'] = seconds
    with open(os.path.join(run_dir, 'params.txt'), 'w') as f:
        for key, value in params.items():
            f.write(f'{key} = {value}\n')
    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        json.dump({k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                   for k, v in params.items()}, f, indent=2)


def write_summary(run_dir, lines):
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def iterate_numbers(est_obj):
    """Four numbers that pin a complex iterate: Frobenius norm, real and
    imaginary parts of the complex sum, and the maximum magnitude."""
    return {'frobenius': float(np.linalg.norm(est_obj)),
            'sum_real': float(np.real(est_obj).sum()),
            'sum_imag': float(np.imag(est_obj).sum()),
            'max_abs': float(np.abs(est_obj).max())}


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
