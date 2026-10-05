"""Compare xptycho with a stored baseline of the old ptycho_pmace code.

A baseline folder (written by the capture_*.py scripts) holds the old
code's inputs and its object at chosen iterations.  This script runs
xptycho from the same inputs and prints, at each stored iteration, the
relative difference between the two objects and the two data errors.

Run in the xptycho environment:
    python compare_baseline.py <baseline folder> [device,device,...] [frames .npy]
An empty device argument ("") uses the default devices.  The frames file is
used when the baseline folder has no inputs/y_meas.npy.
"""
import glob
import json
import os
import sys
import time

import numpy as np

import xptycho as xpt
from xptycho.pmace import Run

# ------------------------------- Parameters -------------------------------
BASELINE = sys.argv[1]
DEVICES = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] else None
FRAMES = sys.argv[3] if len(sys.argv) > 3 else None
PIXEL_PITCH = 1e-8        # meters; any value, the old code works in pixels
WAVELENGTH = 1e-10
DISTANCE = 1.0
# --------------------------------------------------------------------------


def load(name):
    path = os.path.join(BASELINE, 'inputs', name + '.npy')
    return np.load(path) if os.path.exists(path) else None


def main():
    params = json.load(open(os.path.join(BASELINE, 'params.json')))
    for key in ('add_mode', 'orthogonalize_modes'):
        if isinstance(params.get(key), str):           # the captures store lists as text
            params[key] = json.loads(params[key])
    y, bounds, init_obj = load('y_meas'), load('patch_bounds'), load('init_obj')
    if y is None:
        y = np.load(FRAMES)
    ref_obj, window = load('ref_obj'), load('recon_win')
    blind = bool(params.get('joint_recon'))
    probe = load('init_probe') if blind else load('ref_probe')
    n = y.shape[-1]

    probe_positions = (bounds[:, [0, 2]] + n // 2) * PIXEL_PITCH
    det_pixel_pitch = WAVELENGTH * DISTANCE / (n * PIXEL_PITCH)
    if blind and params.get('add_mode'):
        # Mode addition uses the true wavelength and pixel pitch.
        global_pixel, wavelength = params['sampling_interval'], params['wavelength']
        probe_positions = (bounds[:, [0, 2]] + n // 2) * global_pixel
        det_pixel_pitch = wavelength * DISTANCE / (n * global_pixel)
    else:
        wavelength = WAVELENGTH
    model = xpt.PtychoModel(wavelength=wavelength, det_distance=DISTANCE, det_pixel_pitch=det_pixel_pitch,
                            frame_size=n, probe_positions=probe_positions,
                            num_probe_modes=params.get('num_modes_final', 1 if probe.ndim == 2 else len(probe)))
    print('pixel pitch {:.3g} m, wavelength {:.3g} m'.format(model.sample_pixel_pitch, model.wavelength))
    model.set_params(object_shape=init_obj.shape, object_origin=(0.0, 0.0),
                     object_data_fit=params['obj_data_fit_prm'], relaxation=params['rho'],
                     probe_weight_exponent=params['probe_exp'])
    if blind:
        model.set_params(probe_data_fit=params['probe_data_fit_prm'], mode_schedule=params.get('add_mode', []),
                         mode_energy_fraction=params.get('energy_ratio', 0.05),
                         probe_fresnel_radius_pixels=np.sqrt(model.wavelength * params.get('propagation_dist', 0.0))
                         / model.sample_pixel_pitch)
    if DEVICES:
        model.configure_devices(devices=DEVICES)

    layout, starts = model._layout()
    batch = min(model._batch_size(d) for d in layout.devices)
    modes = model._check_probe(probe)
    run = Run(layout, starts, y.astype(np.float32), modes, init_obj, dict(model._recon), blind, batch)
    schedule = sorted(model._recon['mode_schedule'])
    old_errors = np.load(os.path.join(BASELINE, 'curves', 'nrmse_meas.npy'))
    stored = {int(os.path.basename(f)[-7:-4]): f for f in glob.glob(os.path.join(BASELINE, 'iterates', 'est_obj_iter_*.npy'))}
    print('{}: {} frames of {}, object {}, devices {}, batch {}, {}'.format(
        os.path.basename(BASELINE.rstrip('/')), len(y), n, init_obj.shape, [str(d) for d in layout.devices], batch,
        'blind' if blind else 'known probe'))

    start = time.time()
    for iteration in range(1, params['iterations'] + 1):
        run.update_object()
        if blind:
            if iteration in schedule:
                run.add_mode(model._recon['probe_fresnel_radius_pixels'])
                if iteration in params.get('orthogonalize_modes', []):
                    run.orthogonalize_modes()
            run.update_probe()
        if iteration in stored:
            mine, old = run.object(), np.load(stored[iteration])
            mask = window if window is not None else np.ones(mine.shape, dtype=np.float32)
            # The old code stored some iterates scaled to the truth and some not; one complex
            # scale is removed before comparing, inside the old code's window.
            mine = xpt.match_scale(mine * mask, old * mask)
            difference = np.linalg.norm(mine - old * mask) / np.linalg.norm(old * mask)
            error = run.data_error()
            line = 'iteration {:4d}   object difference {:.2e}   data error {:.6f} (old {:.6f})'.format(
                iteration, difference, error, old_errors[iteration - 1])
            if ref_obj is not None:
                line += '   NRMSE to truth {:.6f}'.format(xpt.nrmse(run.object() * mask, ref_obj * mask))
            print(line + '   {:.1f} s'.format(time.time() - start))
    if blind:
        old_modes = np.load(os.path.join(BASELINE, 'iterates', 'probe_modes_iter_{:03d}.npy'.format(params['iterations'])))
        mine = run.probe()
        print('probe difference {:.2e}'.format(np.linalg.norm(mine - old_modes) / np.linalg.norm(old_modes)))


main()
