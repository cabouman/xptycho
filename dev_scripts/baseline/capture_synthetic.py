"""Baselines B1 and B2: the OLD code on the paper's synthetic dataset, known probe.

Mirrors demo/demo_pmace.py of ptycho_pmace: the SyntheticImg ground
truth, the frames at photon peak 1e4 and probe spacing 68, the
comparison window [112, 912]^2, 100 iterations.  B1 uses the paper's
data-fit weight 0.7 (expected NRMSE 0.037); B2 uses the code's default
0.5, so a parameter mistake in the port can be told from a port
mistake.  Iterations 1, 2, and 3 are captured by separate short runs,
since the loop has no randomness.

Run in the pmace_repro environment:
    python capture_synthetic.py [output root] [data root]
"""
import os
import sys

import numpy as np
import pandas as pd

from baseline_common import (import_pmace, make_run_dir, save_inputs, write_params,
                             write_summary, iterate_numbers, Timer)

# ------------------------------- Parameters -------------------------------
OUTPUT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/Documents/GitHub/-archive/ptycho_pmace_baseline_2026-09')
DATA_ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUTPUT_ROOT, 'data')
SYN = os.path.join(DATA_ROOT, 'PMACE_demo_data/synthetic_data/SyntheticImg_data')
OBJ_FILE = os.path.join(SYN, 'ground_truth_img/SIM-generating_image.tiff')
PROBE_FILE = os.path.join(SYN, 'ground_truth_img/SIM-generating_probe.tiff')
DATA_DIR = os.path.join(SYN, 'simulated_data/photon_peak_1e4/probe_dist_68')
WINDOW = [112, 912, 112, 912]
ITERATIONS = 100
SHORT_RUNS = (1, 2, 3)
RHO = 0.5
PROBE_EXP = 1.5
RUNS = {'B1_synthetic_alpha0.7': 0.7, 'B2_synthetic_alpha0.5': 0.5}
PMACE_DIR = os.path.expanduser('~/Documents/GitHub/ptycho_pmace')
# --------------------------------------------------------------------------


def load_data(pu):
    ref_obj = pu.load_img(OBJ_FILE)
    ref_probe = pu.load_img(PROBE_FILE)
    y_meas = pu.load_measurement(os.path.join(DATA_DIR, 'frame_data/'))
    table = pd.read_csv(os.path.join(DATA_DIR, 'Translations.tsv.txt'), sep=None,
                        engine='python', header=0)
    scan_loc = table[['FCx', 'FCy']].to_numpy()
    patch_bounds = pu.get_proj_coords_from_data(scan_loc, y_meas)
    init_obj = pu.gen_init_obj(y_meas, patch_bounds, ref_obj.shape, ref_probe=ref_probe)
    recon_win = np.zeros(init_obj.shape)
    xmin, xmax, ymin, ymax = WINDOW
    recon_win[xmin:xmax, ymin:ymax] = 1
    return ref_obj, ref_probe, y_meas, scan_loc, patch_bounds, init_obj, recon_win


def run_once(pm, inputs, alpha, num_iter, save_dir):
    ref_obj, ref_probe, y_meas, scan_loc, patch_bounds, init_obj, recon_win = inputs
    os.makedirs(save_dir, exist_ok=True)
    return pm.pmace_recon(y_meas, patch_bounds, init_obj, ref_obj=ref_obj, ref_probe=ref_probe,
                          num_iter=num_iter, joint_recon=False, recon_win=recon_win,
                          save_dir=save_dir + '/', obj_data_fit_prm=alpha, rho=RHO,
                          probe_exp=PROBE_EXP, add_reg=False)


def main():
    pm, pu, pn = import_pmace()
    np.random.seed(0)
    inputs = load_data(pu)
    ref_obj, ref_probe, y_meas, scan_loc, patch_bounds, init_obj, recon_win = inputs
    print(f'{len(y_meas)} frames of {y_meas.shape[1]}x{y_meas.shape[2]}, object {ref_obj.shape}')

    for name, alpha in RUNS.items():
        run_dir = make_run_dir(OUTPUT_ROOT, name)
        save_inputs(run_dir, ref_obj=ref_obj, ref_probe=ref_probe, y_meas=y_meas,
                    scan_loc=scan_loc, patch_bounds=patch_bounds, init_obj=init_obj,
                    recon_win=recon_win)
        with Timer() as timer:
            for n in SHORT_RUNS:
                result = run_once(pm, inputs, alpha, n,
                                  os.path.join(run_dir, 'old_code_output', f'iter_{n:03d}'))
                est = np.asarray(result['object'], dtype=np.complex64)
                np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{n:03d}.npy'), est)
            result = run_once(pm, inputs, alpha, ITERATIONS,
                              os.path.join(run_dir, 'old_code_output', 'full'))
        est = np.asarray(result['object'], dtype=np.complex64)
        np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{ITERATIONS:03d}.npy'), est)
        # The every-20 iterates the old code wrote as 4-plane TIFFs.
        for n in range(20, ITERATIONS, 20):
            tiff = os.path.join(run_dir, 'old_code_output', 'full', f'est_obj_iter_{n}.tiff')
            if os.path.exists(tiff):
                np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{n:03d}.npy'),
                        pu.load_img(tiff).astype(np.complex64))
        np.save(os.path.join(run_dir, 'curves', 'nrmse_obj.npy'), np.asarray(result['err_obj']))
        np.save(os.path.join(run_dir, 'curves', 'nrmse_meas.npy'), np.asarray(result['err_meas']))
        numbers = iterate_numbers(est)
        params = dict(dataset='PMACE_demo_data SyntheticImg photon_peak_1e4 probe_dist_68',
                      num_frames=len(y_meas), frame_shape=list(y_meas.shape[1:]),
                      object_shape=list(ref_obj.shape), window=WINDOW, iterations=ITERATIONS,
                      obj_data_fit_prm=alpha, rho=RHO, probe_exp=PROBE_EXP, joint_recon=False,
                      seed=0, **numbers)
        write_params(run_dir, params, PMACE_DIR, timer.seconds)
        write_summary(run_dir, [f'final nrmse_obj  {result["err_obj"][-1]:.6f}',
                                f'final nrmse_meas {result["err_meas"][-1]:.6f}',
                                f'wall time {timer.seconds:.1f} s'])
        print(f'{name}: nrmse_obj {result["err_obj"][-1]:.6f} in {timer.seconds:.0f} s')


if __name__ == '__main__':
    main()
