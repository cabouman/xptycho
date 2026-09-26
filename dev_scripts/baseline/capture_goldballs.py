"""Baseline B3: the OLD code on the preprocessed gold-ball data, known probe.

Mirrors paper_TCI2023/tests/real_data_experiment/recon_GoldBalls_sample.py
of ptycho_pmace_papers, run against the head of ptycho_pmace: the
preprocessed frames from PMACE_demo_data multiplied by the 2-D Tukey
window, positions offset by half a frame, an all-ones initial object,
the reference probe held fixed, alpha 0.1, rho 0.5, probe_exp 1.5, 100
iterations, comparison window [100, 500]^2.  There is no ground truth;
the old output itself is the reference.

Run in the pmace_repro environment:
    python capture_goldballs.py [output root] [data root]
"""
import math
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
DATA_DIR = os.path.join(DATA_ROOT, 'PMACE_demo_data/real_data/GoldBalls_data/processed_GoldBalls_data')
WINDOW = [100, 500, 100, 500]
PHASE_WINDOW = [185, 250, 140, 245]
ITERATIONS = 100
ALPHA = 0.1
RHO = 0.5
PROBE_EXP = 1.5
TUKEY_SHAPE = 0.5
PMACE_DIR = os.path.expanduser('~/Documents/GitHub/ptycho_pmace')
# --------------------------------------------------------------------------


def main():
    pm, pu, pn = import_pmace()
    np.random.seed(0)
    ref_probe = pu.load_img(os.path.join(DATA_DIR, 'ref_probe.tiff'))
    y_raw = pu.load_measurement(os.path.join(DATA_DIR, 'frame_data/'))
    tukey = pu.gen_tukey_2D_window(np.zeros_like(y_raw[0]), TUKEY_SHAPE)
    y_meas = y_raw * tukey
    table = pd.read_csv(os.path.join(DATA_DIR, 'Translations.tsv.txt'), sep=None,
                        engine='python', header=0)
    scan_loc = table[['FCx', 'FCy']].to_numpy()
    patch_bounds = pu.get_proj_coords_from_data(scan_loc + y_meas.shape[1] / 2, y_meas)
    img_sz = math.ceil(np.amax(scan_loc + np.maximum(y_meas.shape[1], y_meas.shape[2])))
    init_obj = np.ones((img_sz, img_sz), dtype=np.complex64)
    recon_win = np.zeros(init_obj.shape)
    xmin, xmax, ymin, ymax = WINDOW
    recon_win[xmin:xmax, ymin:ymax] = 1
    print(f'{len(y_meas)} frames of {y_meas.shape[1]}x{y_meas.shape[2]}, object {init_obj.shape}')

    run_dir = make_run_dir(OUTPUT_ROOT, 'B3_goldballs_known_probe')
    save_inputs(run_dir, ref_probe=ref_probe, y_raw=y_raw, tukey=tukey, y_meas=y_meas,
                scan_loc=scan_loc, patch_bounds=patch_bounds, init_obj=init_obj,
                recon_win=recon_win)
    save_dir = os.path.join(run_dir, 'old_code_output', 'full') + '/'
    os.makedirs(save_dir, exist_ok=True)
    with Timer() as timer:
        result = pm.pmace_recon(y_meas, patch_bounds, init_obj, ref_probe=ref_probe,
                                num_iter=ITERATIONS, joint_recon=False, recon_win=recon_win,
                                save_dir=save_dir, obj_data_fit_prm=ALPHA, rho=RHO,
                                probe_exp=PROBE_EXP, add_reg=False)
    est = np.asarray(result['object'], dtype=np.complex64)
    np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{ITERATIONS:03d}.npy'), est)
    for n in range(20, ITERATIONS, 20):
        tiff = os.path.join(save_dir, f'est_obj_iter_{n}.tiff')
        if os.path.exists(tiff):
            np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{n:03d}.npy'),
                    pu.load_img(tiff).astype(np.complex64))
    np.save(os.path.join(run_dir, 'curves', 'nrmse_meas.npy'), np.asarray(result['err_meas']))
    params = dict(dataset='PMACE_demo_data processed_GoldBalls_data', num_frames=len(y_meas),
                  frame_shape=list(y_meas.shape[1:]), object_shape=list(init_obj.shape),
                  window=WINDOW, phase_window=PHASE_WINDOW, tukey_shape=TUKEY_SHAPE,
                  iterations=ITERATIONS, obj_data_fit_prm=ALPHA, rho=RHO, probe_exp=PROBE_EXP,
                  joint_recon=False, init='ones', position_offset='frame_height / 2', seed=0,
                  **iterate_numbers(est))
    write_params(run_dir, params, PMACE_DIR, timer.seconds)
    write_summary(run_dir, [f'final nrmse_meas {result["err_meas"][-1]:.6f}',
                            f'wall time {timer.seconds:.1f} s'])
    print(f'B3: nrmse_meas {result["err_meas"][-1]:.6f} in {timer.seconds:.0f} s')


if __name__ == '__main__':
    main()
