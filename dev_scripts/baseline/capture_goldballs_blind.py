"""Baseline B7: the OLD code on the gold-ball data, probe estimated, two modes.

Mirrors AuBalls_700ms_30nmStep_3_full_recon_with_two_mode.py of
ptycho_pmace_papers (paper_BM-PMACE2024): the 794 preprocessed frames with
the Tukey window, positions from the translations recorded in the raw file
(not flipped, shifted so the smallest is 10 pixels), an object of ones, the
initial probe without Fresnel propagation, then 100 iterations with the
second mode added at iteration 10, 5 percent of the energy, and the modes
made orthogonal at that iteration.

Run in the pmace_repro environment:
    python capture_goldballs_blind.py <raw .cxi file> [output root]
"""
import math
import os
import sys

import h5py
import numpy as np

from baseline_common import import_pmace, make_run_dir, save_inputs, write_params, write_summary, Timer

# ------------------------------- Parameters -------------------------------
RAW_FILE = sys.argv[1]
OUTPUT_ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
    '~/Documents/GitHub/-archive/ptycho_pmace_baseline_2026-09')
FRAMES = os.path.join(OUTPUT_ROOT, 'B3_goldballs_known_probe', 'inputs', 'y_meas.npy')   # the same 794 frames
OUTLIERS = [581, 648, 649, 723, 763, 764]
NAME = 'B7_goldballs_blind_two_modes'
ITERATIONS = 100
ARGS = dict(obj_data_fit_prm=0.5, probe_data_fit_prm=0.6, probe_exp=1.25, rho=0.5,
            add_mode=[10], energy_ratio=0.05, orthogonalize_modes=[10])
# Mode addition uses the defaults of pmace_recon, which the paper's script does not override.
IMG_PX_SZ, WAVELENGTH, PROPAGATION_DIST = 4.52e-9, 1.24e-9, 1e-7
PMACE_DIR = os.path.expanduser('~/Documents/GitHub/ptycho_pmace')
# --------------------------------------------------------------------------


def main():
    pm, pu, pn = import_pmace()
    np.random.seed(0)
    y_meas = np.load(FRAMES)
    with h5py.File(RAW_FILE, 'r') as f:
        trans = np.array(f['entry_1/data_1/translation'])
        distance = float(np.array(f['entry_1/instrument_1/detector_1/distance']))
        x_pixel = float(np.array(f['entry_1/instrument_1/detector_1/x_pixel_size']))
        energy_ev = float(np.array(f['entry_1/instrument_1/source_1/energy'])) * 6.241509e18
    wavelength_nm = 1239.84193 / energy_ev
    trans = np.delete(trans, OUTLIERS, axis=0)[:, :2]
    img_pixel = wavelength_nm * 1e-9 * distance / (y_meas.shape[-1] * x_pixel)
    trans_px = trans / img_pixel
    scan_loc = trans_px - np.min(trans_px) + 10
    patch_bounds = pu.get_proj_coords_from_data(scan_loc + y_meas.shape[-1] / 2, y_meas)
    img_sz = math.ceil(np.max(scan_loc + np.maximum(y_meas.shape[1], y_meas.shape[2])))
    ones = np.ones((img_sz, img_sz), dtype=np.complex64)
    # The values the paper's configuration file gives; without propagation they change nothing.
    init_probe = pu.gen_init_probe(y_meas, patch_bounds, ones, fres_propagation=False, sampling_interval=4.52e-9,
                                   source_wl=1.24e-9, propagation_dist=3e-7)
    init_obj = pu.gen_init_obj(y_meas, patch_bounds, ones.shape, ref_probe=init_probe)
    print(f'{len(y_meas)} frames of {y_meas.shape[1]}x{y_meas.shape[2]}, object {init_obj.shape}, pixel {img_pixel:.4g} m')

    run_dir = make_run_dir(OUTPUT_ROOT, NAME)
    save_inputs(run_dir, scan_loc=scan_loc, patch_bounds=patch_bounds, init_obj=init_obj, init_probe=init_probe)
    save_dir = os.path.join(run_dir, 'old_code_output', 'full') + '/'
    os.makedirs(save_dir, exist_ok=True)
    with Timer() as timer:
        # The old code updates the starting probe array in place, so the run gets a copy.
        result = pm.pmace_recon(y_meas, patch_bounds, init_obj, init_probe=init_probe.copy(),
                                num_iter=ITERATIONS, joint_recon=True, recon_win=None, save_dir=save_dir,
                                add_reg=False, img_px_sz=IMG_PX_SZ, wavelength=WAVELENGTH,
                                propagation_dist=PROPAGATION_DIST, **ARGS)
    est = np.asarray(result['object'], dtype=np.complex64)
    modes = np.asarray(result['probe'], dtype=np.complex64)
    np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{ITERATIONS:03d}.npy'), est)
    np.save(os.path.join(run_dir, 'iterates', f'probe_modes_iter_{ITERATIONS:03d}.npy'), modes)
    for n in range(10, ITERATIONS, 10):
        tiff = os.path.join(save_dir, f'est_obj_iter_{n}.tiff')
        if os.path.exists(tiff):
            np.save(os.path.join(run_dir, 'iterates', f'est_obj_iter_{n:03d}.npy'), pu.load_img(tiff).astype(np.complex64))
    np.save(os.path.join(run_dir, 'curves', 'nrmse_meas.npy'), np.asarray(result['err_meas']))
    params = dict(dataset='gold balls, 794 preprocessed frames with the Tukey window', num_frames=len(y_meas),
                  frame_shape=list(y_meas.shape[1:]), object_shape=list(init_obj.shape), iterations=ITERATIONS,
                  joint_recon=True, sampling_interval=IMG_PX_SZ, wavelength=WAVELENGTH,
                  propagation_dist=PROPAGATION_DIST, true_pixel=img_pixel, seed=0, num_modes_final=len(modes),
                  init='gen_init_obj', **{k: (v if not isinstance(v, list) else str(v)) for k, v in ARGS.items()})
    write_params(run_dir, params, PMACE_DIR, timer.seconds)
    write_summary(run_dir, [f'final nrmse_meas {result["err_meas"][-1]:.6f}', f'wall time {timer.seconds:.1f} s'])
    print(f'{NAME}: nrmse_meas {result["err_meas"][-1]:.6f} in {timer.seconds:.0f} s')


if __name__ == '__main__':
    main()
