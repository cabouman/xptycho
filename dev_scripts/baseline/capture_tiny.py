"""Baseline B5: a tiny noiseless PMACE run of the OLD code, every iterate saved.

A 64 x 64 complex object and a 16 x 16 probe built from fixed formulas,
25 scan positions on a 5 x 5 grid at 12 pixel spacing, no noise, known
probe.  pmace_recon is called once per iteration count from 1 to 20;
the loop has no randomness, so each call gives the exact iterate at
that index.  The run writes every iterate, the error curves, and a JSON
file of four numbers per iterate that the xptycho fast test suite
compares against.

Run in the pmace_repro environment:
    python capture_tiny.py [output root]
"""
import json
import os
import shutil
import sys

import numpy as np

from baseline_common import (import_pmace, make_run_dir, save_inputs, write_params,
                             write_summary, iterate_numbers, Timer)

# ------------------------------- Parameters -------------------------------
OBJECT_SIZE = 64
PROBE_SIZE = 16
GRID = 5
SPACING = 12
ITERATIONS = 20
OBJ_DATA_FIT = 0.5          # the old code's default alpha
RHO = 0.5
PROBE_EXP = 1.5             # the old code's default kappa
OUTPUT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/Documents/GitHub/-archive/ptycho_pmace_baseline_2026-09')
PMACE_DIR = os.path.expanduser('~/Documents/GitHub/ptycho_pmace')
# --------------------------------------------------------------------------


def make_object(n):
    """A smooth complex transmittance: magnitude 0.85 to 1.0, phase -0.6 to 0."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    magnitude = 1.0 - 0.15 * (0.5 + 0.5 * np.sin(2 * np.pi * (x + 0.3 * y))) * np.exp(-((x - 0.5) ** 2 + (y - 0.4) ** 2) / 0.12)
    phase = -0.6 * np.exp(-((x - 0.55) ** 2 + (y - 0.5) ** 2) / 0.08) - 0.2 * x
    return (magnitude * np.exp(1j * phase)).astype(np.complex64)


def make_probe(m):
    """A Gaussian amplitude with a quadratic phase, a focused beam."""
    y, x = np.mgrid[0:m, 0:m] - (m - 1) / 2
    r2 = (x ** 2 + y ** 2) / (0.18 * m) ** 2
    return (np.exp(-r2) * np.exp(1j * 0.6 * r2)).astype(np.complex64)


def make_patch_bounds(n, m, grid, spacing):
    """Integer patch bounds [row_start, row_end, col_start, col_end] on a
    centered grid; every patch lies inside the object."""
    centers = n / 2 + (np.arange(grid) - (grid - 1) / 2) * spacing
    bounds = []
    for r in centers:
        for c in centers:
            r0, c0 = int(round(r)) - m // 2, int(round(c)) - m // 2
            bounds.append([r0, r0 + m, c0, c0 + m])
    return np.asarray(bounds, dtype=int)


def main():
    pm, pu, pn = import_pmace()
    np.random.seed(0)
    obj = make_object(OBJECT_SIZE)
    probe = make_probe(PROBE_SIZE)
    patch_bounds = make_patch_bounds(OBJECT_SIZE, PROBE_SIZE, GRID, SPACING)
    intensity = pu.gen_syn_data(obj, probe, patch_bounds, add_noise=False)
    y_meas = np.sqrt(np.asarray(intensity, dtype=np.float32))
    init_obj = pu.gen_init_obj(y_meas, patch_bounds, obj.shape, probe)
    recon_win = np.ones(obj.shape, dtype=np.float32)

    run_dir = make_run_dir(OUTPUT_ROOT, 'B5_tiny')
    save_inputs(run_dir, ref_obj=obj, ref_probe=probe, patch_bounds=patch_bounds,
                y_meas=y_meas, init_obj=init_obj, recon_win=recon_win)

    records = {}
    with Timer() as timer:
        for n in range(1, ITERATIONS + 1):
            save_dir = os.path.join(run_dir, 'old_code_output', f'iter_{n:02d}') + '/'
            os.makedirs(save_dir, exist_ok=True)
            result = pm.pmace_recon(y_meas, patch_bounds, init_obj, ref_obj=obj,
                                    ref_probe=probe, num_iter=n, joint_recon=False,
                                    recon_win=recon_win, save_dir=save_dir,
                                    obj_data_fit_prm=OBJ_DATA_FIT, rho=RHO,
                                    probe_exp=PROBE_EXP, add_reg=False)
            est = np.asarray(result['object'], dtype=np.complex64)
            np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{n:02d}.npy'), est)
            numbers = iterate_numbers(est)
            numbers['nrmse_obj'] = float(result['err_obj'][-1])
            numbers['nrmse_meas'] = float(result['err_meas'][-1])
            records[str(n)] = numbers
            print(f'iteration {n:2d}: nrmse_obj {numbers["nrmse_obj"]:.6f} '
                  f'nrmse_meas {numbers["nrmse_meas"]:.6f}')
            if n == ITERATIONS:
                np.save(os.path.join(run_dir, 'curves', 'nrmse_obj.npy'), np.asarray(result['err_obj']))
                np.save(os.path.join(run_dir, 'curves', 'nrmse_meas.npy'), np.asarray(result['err_meas']))
            else:
                shutil.rmtree(save_dir)

    tiny = {'description': 'Old ptycho_pmace code, tiny noiseless run, known probe; '
                           'see dev_scripts/baseline/capture_tiny.py',
            'object_size': OBJECT_SIZE, 'probe_size': PROBE_SIZE, 'grid': GRID,
            'spacing': SPACING, 'obj_data_fit_prm': OBJ_DATA_FIT, 'rho': RHO,
            'probe_exp': PROBE_EXP, 'iterates': records}
    with open(os.path.join(run_dir, 'tiny_baseline.json'), 'w') as f:
        json.dump(tiny, f, indent=2)

    params = dict(object_size=OBJECT_SIZE, probe_size=PROBE_SIZE, grid=GRID, spacing=SPACING,
                  iterations=ITERATIONS, obj_data_fit_prm=OBJ_DATA_FIT, rho=RHO,
                  probe_exp=PROBE_EXP, noise='none', joint_recon=False, seed=0)
    write_params(run_dir, params, PMACE_DIR, timer.seconds)
    write_summary(run_dir, [f'final nrmse_obj  {records[str(ITERATIONS)]["nrmse_obj"]:.6f}',
                            f'final nrmse_meas {records[str(ITERATIONS)]["nrmse_meas"]:.6f}',
                            f'wall time {timer.seconds:.1f} s'])
    print('wrote', run_dir)


if __name__ == '__main__':
    main()
