"""Write the reference file of the gold-ball demo from the reference data of
ptycho_pmace, and check the preprocessing against its frames.

The file holds one sample: the known probe that came with PMACE_demo_data
(estimated with another program), and the object ptycho_pmace reconstructs
with it in 100 iterations (baseline B3).  Its origin places the object on
the positions recorded in the raw file, so a reconstruction from the raw
file can use the same grid.

The measured data is the Ptychography Gold Ball Example Dataset of Stefano
Marchesini et al., CXIDB ID 65, doi:10.11577/1454414 (CC0).

Usage:  python make_goldballs_demo_data.py <B3 baseline folder> <raw .cxi file> <output file>
"""
import os
import sys

import numpy as np

import xptycho as xpt
import xptycho.preprocess as xpp

BASELINE, RAW_FILE, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3]
FRAME_SIZE, TUKEY_SHAPE, OUTLIER_THRESHOLD = 512, 0.5, 2.0


def load(name):
    return np.load(os.path.join(BASELINE, 'inputs', name + '.npy'))


# The preprocessing of demo 3, on the raw file.
raw = xpp.cxi.load_raw(RAW_FILE)
frames = xpp.subtract_dark(raw['frames'], raw['dark_frames'])
outlier = xpp.find_outlier_frames(frames, OUTLIER_THRESHOLD)
frames, translations = frames[~outlier], raw['translations'][~outlier]
frames = xpp.crop_frames(frames, xpp.diffraction_center(frames), FRAME_SIZE)
frames = frames * xpp.tukey_window(FRAME_SIZE, TUKEY_SHAPE) ** 2
positions = -translations[:, [1, 0]]

# Check: the frames equal those ptycho_pmace reconstructs.
reference_frames = load('y_meas')
worst = max(float(np.abs(np.sqrt(frames[j].astype(np.float64)) - reference_frames[j]).max()) for j in range(len(frames)))
print('outliers removed: {}'.format(np.flatnonzero(outlier).tolist()))
print('largest amplitude difference from the reference frames: {:.2e} of {:.1f}'.format(worst, reference_frames.max()))

# The reference object is on a grid of its own.  Its origin is placed so that the patch
# centers of the reference run fall, on average, on the positions recorded in the raw file.
pixel_pitch = raw['wavelength'] * raw['detector_distance'] / (FRAME_SIZE * raw['detector_pitch'])
bounds = load('patch_bounds')
centers = bounds[:, [0, 2]] + FRAME_SIZE // 2                     # pixels of the reference object
offset = (centers - positions / pixel_pitch).mean(axis=0)
origin = -offset * pixel_pitch
residual = centers - (positions - origin) / pixel_pitch
print('reference patch centers minus recorded positions: rms {:.2f} pixels, largest {:.2f}'.format(
    np.sqrt((residual ** 2).mean()), np.abs(residual).max()))

reference = xpt.Sample(np.load(os.path.join(BASELINE, 'iterates', 'est_obj_iter_100.npy')), load('ref_probe'),
                       pixel_pitch, origin=origin,
                       name='reference for the gold balls of S. Marchesini et al., CXIDB ID 65: the known probe, and the '
                            'object ptycho_pmace reconstructs with it in 100 iterations')
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)
reference.save(OUTPUT)
print(reference.summary())
print('origin (m): {}'.format(reference.origin))
print('{}: {:.1f} MB'.format(OUTPUT, os.path.getsize(OUTPUT) / 1e6))
