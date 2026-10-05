"""Baseline B4: the OLD code on the blind two-mode synthetic dataset.

Mirrors demo/demo_blind_multi_mode_pmace.py of ptycho_pmace: the
BMPMACE_demo_data ground truth (object and two probe modes), 400
frames, the comparison window [240, 840]^2, the Fresnel-propagated
initial probe, then two runs of 200 iterations: single mode
(alpha1 0.6, alpha2 0.6, kappa 1.25) and two modes (alpha1 0.5, alpha2
0.6, kappa 1.25, second mode added at iteration 20 with energy ratio
0.1).

Run in the pmace_repro environment:
    python capture_blind.py [output root] [data root]
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
DATA_DIR = os.path.join(DATA_ROOT, 'pmace-data/BMPMACE_demo_data')
WINDOW = [240, 840, 240, 840]
ITERATIONS = 200
SAMPLING_INTERVAL = 2.4e-9      # m
WAVELENGTH = 1.4e-9             # m
PROPAGATION_DIST = 2e-6         # m
RHO = 0.5
RUNS = {
    'B4a_blind_single_mode': dict(obj_data_fit_prm=0.6, probe_data_fit_prm=0.6, probe_exp=1.25,
                                  add_mode=[]),
    'B4b_blind_two_modes': dict(obj_data_fit_prm=0.5, probe_data_fit_prm=0.6, probe_exp=1.25,
                                add_mode=[20], energy_ratio=0.1, wavelength=WAVELENGTH,
                                propagation_dist=PROPAGATION_DIST, img_px_sz=SAMPLING_INTERVAL),
}
PMACE_DIR = os.path.expanduser('~/Documents/GitHub/ptycho_pmace')
# --------------------------------------------------------------------------


def main():
    pm, pu, pn = import_pmace()
    np.random.seed(0)
    stored = os.path.join(OUTPUT_ROOT, 'B4a_blind_single_mode', 'inputs')
    if not os.path.isdir(DATA_DIR) and os.path.isdir(stored):
        # The dataset is not on disk: use the inputs the single-mode capture stored.
        load = lambda name: np.load(os.path.join(stored, name + '.npy'))
        ref_obj, ref_modes = load('ref_obj'), [load('ref_probe_mode_0'), load('ref_probe_mode_1')]
        y_meas, scan_loc, patch_bounds = load('y_meas'), load('scan_loc'), load('patch_bounds')
        recon_win, init_probe, init_obj = load('recon_win'), load('init_probe'), load('init_obj')
    else:
        ref_obj = pu.load_img(os.path.join(DATA_DIR, 'ground_truth_img/ref_object.tiff'))
        ref_modes = [pu.load_img(os.path.join(DATA_DIR, f'ground_truth_img/ref_probe_mode_{k}.tiff'))
                     for k in (0, 1)]
        y_meas = pu.load_measurement(os.path.join(DATA_DIR, 'frame_data/'))
        table = pd.read_csv(os.path.join(DATA_DIR, 'Translations.tsv.txt'), sep=None,
                            engine='python', header=0)
        scan_loc = table[['FCx', 'FCy']].to_numpy()
        patch_bounds = pu.get_proj_coords_from_data(scan_loc, y_meas)
        recon_win = np.zeros(ref_obj.shape)
        xmin, xmax, ymin, ymax = WINDOW
        recon_win[xmin:xmax, ymin:ymax] = 1
        ones = np.ones_like(ref_obj, dtype=np.complex64)
        init_probe = pu.gen_init_probe(y_meas, patch_bounds, ones, fres_propagation=True,
                                       sampling_interval=SAMPLING_INTERVAL, source_wl=WAVELENGTH,
                                       propagation_dist=PROPAGATION_DIST)
        init_obj = pu.gen_init_obj(y_meas, patch_bounds, ones.shape, ref_probe=init_probe)
    print(f'{len(y_meas)} frames of {y_meas.shape[1]}x{y_meas.shape[2]}, object {ref_obj.shape}')

    names = sys.argv[3].split(',') if len(sys.argv) > 3 else list(RUNS)   # optional: which runs
    for name in names:
        args = RUNS[name]
        run_dir = make_run_dir(OUTPUT_ROOT, name)
        save_inputs(run_dir, ref_obj=ref_obj, ref_probe_mode_0=ref_modes[0],
                    ref_probe_mode_1=ref_modes[1], y_meas=y_meas, scan_loc=scan_loc,
                    patch_bounds=patch_bounds, init_obj=init_obj, init_probe=init_probe,
                    recon_win=recon_win)
        save_dir = os.path.join(run_dir, 'old_code_output', 'full') + '/'
        os.makedirs(save_dir, exist_ok=True)
        with Timer() as timer:
            # The old code updates the starting probe array in place, so each run gets a copy.
            result = pm.pmace_recon(y_meas, patch_bounds, init_obj, init_probe=init_probe.copy(),
                                    ref_obj=ref_obj, ref_probe=ref_modes, num_iter=ITERATIONS,
                                    joint_recon=True, recon_win=recon_win, save_dir=save_dir,
                                    rho=RHO, add_reg=False, **args)
        est = np.asarray(result['object'], dtype=np.complex64)
        np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{ITERATIONS:03d}.npy'), est)
        modes = np.asarray(result['probe'], dtype=np.complex64)
        np.save(os.path.join(run_dir, 'iterates', f'probe_modes_iter_{ITERATIONS:03d}.npy'), modes)
        for n in range(20, ITERATIONS, 20):
            tiff = os.path.join(save_dir, f'est_obj_iter_{n}.tiff')
            if os.path.exists(tiff):
                np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{n:03d}.npy'),
                        pu.load_img(tiff).astype(np.complex64))
        np.save(os.path.join(run_dir, 'curves', 'nrmse_obj.npy'), np.asarray(result['err_obj']))
        np.save(os.path.join(run_dir, 'curves', 'nrmse_meas.npy'), np.asarray(result['err_meas']))
        for k, curve in enumerate(result['err_probe']):
            if curve:
                np.save(os.path.join(run_dir, 'curves', f'nrmse_probe_mode_{k}.npy'), np.asarray(curve))
        params = dict(dataset='pmace-data BMPMACE_demo_data', num_frames=len(y_meas),
                      frame_shape=list(y_meas.shape[1:]), object_shape=list(ref_obj.shape),
                      window=WINDOW, iterations=ITERATIONS, joint_recon=True, rho=RHO,
                      sampling_interval=SAMPLING_INTERVAL, wavelength=WAVELENGTH,
                      propagation_dist=PROPAGATION_DIST, seed=0, num_modes_final=len(modes),
                      **{k: (v if not isinstance(v, list) else str(v)) for k, v in args.items()
                         if k not in ('wavelength', 'propagation_dist')},
                      **iterate_numbers(est))
        write_params(run_dir, params, PMACE_DIR, timer.seconds)
        probe_lines = [f'final nrmse_probe_mode_{k} {curve[-1]:.6f}'
                       for k, curve in enumerate(result['err_probe']) if curve]
        write_summary(run_dir, [f'final nrmse_obj  {result["err_obj"][-1]:.6f}',
                                f'final nrmse_meas {result["err_meas"][-1]:.6f}'] + probe_lines +
                               [f'wall time {timer.seconds:.1f} s'])
        print(f'{name}: nrmse_obj {result["err_obj"][-1]:.6f} in {timer.seconds:.0f} s')


if __name__ == '__main__':
    main()
