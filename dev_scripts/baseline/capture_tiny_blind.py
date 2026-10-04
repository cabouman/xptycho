"""Baseline B6: a tiny noiseless blind two-mode PMACE run of the OLD code.

The tiny problem of capture_tiny.py (a 64 x 64 object, a 16 x 16 probe,
49 positions), with data from two probe modes.  The old code starts from
its own initial probe and object, estimates the probe, and adds the
second mode at iteration 2.  pmace_recon is called once per iteration
count (1, 3, 4, 5, 6); the loop has no randomness, so each call gives the
exact iterate.  The run writes a JSON file of numbers per iterate that
the xptycho fast test suite compares against.

Run in the pmace_repro environment:
    python capture_tiny_blind.py [output root]
"""
import json
import os
import shutil
import sys

import numpy as np

from baseline_common import (import_pmace, make_run_dir, save_inputs, write_params,
                             write_summary, iterate_numbers, Timer)
from capture_tiny import make_object, make_probe, make_patch_bounds, OBJECT_SIZE, PROBE_SIZE, GRID, SPACING

# ------------------------------- Parameters -------------------------------
ITERATIONS = 6
ITERATION_COUNTS = [1, 3, 4, 5, 6]   # not 2: the old code fails to save its probe error at exactly 2 iterations
ADD_MODE = [2]                  # the iteration at which the second mode is added
ENERGY_RATIO = 0.05
OBJ_DATA_FIT = 0.5
PROBE_DATA_FIT = 0.6
RHO = 0.5
PROBE_EXP = 1.25
PIXEL_SIZE = 1e-8               # m
WAVELENGTH = 1e-10              # m
PROPAGATION_DIST = 2e-6         # m
SECOND_MODE_SCALE = 0.4         # the second true mode relative to the first
OUTPUT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/Documents/GitHub/-archive/ptycho_pmace_baseline_2026-09')
PMACE_DIR = os.path.expanduser('~/Documents/GitHub/ptycho_pmace')
# --------------------------------------------------------------------------


def make_second_mode(m):
    """The first mode times a phase ramp and an odd amplitude, so the two
    modes differ in shape."""
    y, x = np.mgrid[0:m, 0:m] - (m - 1) / 2
    return (SECOND_MODE_SCALE * make_probe(m) * (x / (0.32 * m)) * np.exp(1j * 0.3 * y)).astype(np.complex64)


def main():
    pm, pu, pn = import_pmace()
    np.random.seed(0)
    obj = make_object(OBJECT_SIZE)
    modes = np.stack([make_probe(PROBE_SIZE), make_second_mode(PROBE_SIZE)])
    patch_bounds = make_patch_bounds(OBJECT_SIZE, PROBE_SIZE, GRID, SPACING)
    intensity = pu.gen_syn_data(obj, modes, patch_bounds, add_noise=False)
    y_meas = np.sqrt(np.asarray(intensity, dtype=np.float32))
    ones = np.ones(obj.shape, dtype=np.complex64)
    init_probe = pu.gen_init_probe(y_meas, patch_bounds, ones, fres_propagation=True,
                                   sampling_interval=PIXEL_SIZE, source_wl=WAVELENGTH,
                                   propagation_dist=PROPAGATION_DIST)
    init_obj = pu.gen_init_obj(y_meas, patch_bounds, obj.shape, ref_probe=init_probe)
    recon_win = np.ones(obj.shape, dtype=np.float32)

    run_dir = make_run_dir(OUTPUT_ROOT, 'B6_tiny_blind')
    save_inputs(run_dir, ref_obj=obj, ref_probe_modes=modes, patch_bounds=patch_bounds, y_meas=y_meas,
                init_obj=init_obj, init_probe=init_probe, recon_win=recon_win)
    records = {}
    with Timer() as timer:
        for n in ITERATION_COUNTS:
            save_dir = os.path.join(run_dir, 'old_code_output', f'iter_{n:02d}') + '/'
            os.makedirs(save_dir, exist_ok=True)
            # The old code updates the starting probe array in place, so each call gets a copy.
            result = pm.pmace_recon(y_meas, patch_bounds, init_obj, init_probe=init_probe.copy(), ref_obj=obj,
                                    ref_probe=list(modes), num_iter=n, joint_recon=True, recon_win=recon_win,
                                    save_dir=save_dir, obj_data_fit_prm=OBJ_DATA_FIT,
                                    probe_data_fit_prm=PROBE_DATA_FIT, rho=RHO, probe_exp=PROBE_EXP,
                                    add_reg=False, add_mode=ADD_MODE, energy_ratio=ENERGY_RATIO,
                                    wavelength=WAVELENGTH, propagation_dist=PROPAGATION_DIST,
                                    img_px_sz=PIXEL_SIZE)
            est = np.asarray(result['object'], dtype=np.complex64)
            probe = np.asarray(result['probe'], dtype=np.complex64)
            np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{n:02d}.npy'), est)
            np.save(os.path.join(run_dir, 'iterates', f'probe_modes_iter_{n:02d}.npy'), probe)
            records[str(n)] = dict(object=iterate_numbers(est), probe=[iterate_numbers(m) for m in probe],
                                   nrmse_meas=float(result['err_meas'][-1]))
            print(f'iteration {n}: {len(probe)} modes, nrmse_meas {records[str(n)]["nrmse_meas"]:.6f}')
            shutil.rmtree(save_dir)

    tiny = {'description': 'Old ptycho_pmace code, tiny noiseless blind two-mode run; '
                           'see dev_scripts/baseline/capture_tiny_blind.py',
            'add_mode': ADD_MODE, 'energy_ratio': ENERGY_RATIO, 'obj_data_fit_prm': OBJ_DATA_FIT,
            'probe_data_fit_prm': PROBE_DATA_FIT, 'rho': RHO, 'probe_exp': PROBE_EXP,
            'pixel_size': PIXEL_SIZE, 'wavelength': WAVELENGTH, 'propagation_dist': PROPAGATION_DIST,
            'second_mode_scale': SECOND_MODE_SCALE,
            'init_probe': iterate_numbers(init_probe), 'init_obj': iterate_numbers(init_obj),
            'iterates': records}
    with open(os.path.join(run_dir, 'tiny_blind_baseline.json'), 'w') as f:
        json.dump(tiny, f, indent=2)
    params = dict(object_size=OBJECT_SIZE, probe_size=PROBE_SIZE, grid=GRID, spacing=SPACING,
                  iterations=ITERATIONS, add_mode=str(ADD_MODE), energy_ratio=ENERGY_RATIO,
                  obj_data_fit_prm=OBJ_DATA_FIT, probe_data_fit_prm=PROBE_DATA_FIT, rho=RHO,
                  probe_exp=PROBE_EXP, noise='none', joint_recon=True, seed=0)
    write_params(run_dir, params, PMACE_DIR, timer.seconds)
    write_summary(run_dir, [f'final nrmse_meas {records[str(ITERATIONS)]["nrmse_meas"]:.6f}',
                            f'wall time {timer.seconds:.1f} s'])
    print('wrote', run_dir)


if __name__ == '__main__':
    main()
