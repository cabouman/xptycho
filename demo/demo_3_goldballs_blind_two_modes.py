"""Demo 3: preprocess and reconstruct measured data, the gold-ball scan.

This demo starts from a raw file, 800 frames of 621 x 621 detector counts
and 20 dark frames, and needs nothing else: the probe is estimated from the
data, with two modes.  It follows the measured-data experiment of the blind
multi-mode PMACE paper (IEEE Transactions on Computational Imaging, 2025).

The data is the Ptychography Gold Ball Example Dataset of Stefano Marchesini
et al., Lawrence Berkeley National Laboratory, collected at beamline 5.3.2.1
of the Advanced Light Source and published in the Coherent X-ray Imaging
Data Bank, cxidb.org, entry 65, doi:10.11577/1454414, under the CC0 public
domain dedication.

The preprocessing, one step at a time:

1. subtract the mean dark frame;
2. remove the frames whose mean intensity is an outlier;
3. find the center of the diffraction patterns and crop about it;
4. multiply by a Tukey window.

The reconstruction starts from one probe mode computed from the data, adds
the second at iteration 10, and makes the two modes orthogonal when it does.
The scan positions are used as recorded; the object then has the orientation
of the SEM image supplied with the data set.

The raw file (about 630 MB) downloads on first run.  Run from the repository
root.  The run takes a few minutes on one GPU.
"""
import matplotlib.pyplot as plt
import numpy as np

import xptycho as xpt
import xptycho.preprocess as xpp

# ---------------------------- Parameters ----------------------------
# The raw file, from the Coherent X-ray Imaging Data Bank (S. Marchesini et al., CXIDB ID 65).
RAW_URL = 'https://cxidb.org/data/65/AuBalls_700ms_30nmStep_3_full.cxi'
# A backup copy of the raw file, to use only if the address above stops working:
# 'https://www.datadepot.rcac.purdue.edu/bouman/data/AuBalls_700ms_30nmStep_3_full.cxi'

DATA_DIR = './demo/input'

# The preprocessing, as in the paper.
OUTLIER_THRESHOLD = 2.0         # standard deviations of the mean frame intensity
FRAME_SIZE = 512                # pixels kept about the diffraction center
TUKEY_SHAPE = 0.5               # fraction of the window inside the taper

# The reconstruction.
NUM_PROBE_MODES = 2
ITERATIONS = 100
OBJECT_DATA_FIT = 0.5           # alpha_1 in the paper
PROBE_DATA_FIT = 0.6            # alpha_2
PROBE_WEIGHT_EXPONENT = 1.25    # kappa
RELAXATION = 0.5                # rho
MODE_SCHEDULE = [10]            # the iteration at which the second mode is added
MODE_ENERGY_FRACTION = 0.05     # the share of the probe energy the new mode starts with
ORTHOGONALIZE_MODES = True      # make the modes orthogonal when a mode is added
MODE_DISTANCE = 9e-7            # m, Fresnel propagation of a new mode

# The part of the object to view: first and last row, first and last column.
# This is 400 x 400 pixels about the center of the scan.
VIEW_REGION = (77, 477, 99, 499)

OUTPUT_DIR = './demo/output/demo_3_goldballs_blind_two_modes'
# --------------------------------------------------------------------

# Read the raw file: detector counts, dark frames, recorded translations, instrument facts.
raw = xpp.cxi.load_raw(xpt.download(RAW_URL, DATA_DIR))
print('raw file: {} frames of {} x {}, {} dark frames'.format(*raw['frames'].shape, len(raw['dark_frames'])))

# 1. Subtract the mean dark frame; negative values become zero.
frames = xpp.subtract_dark(raw['frames'], raw['dark_frames'])

# 2. Remove the frames whose mean intensity is an outlier, and their translations.
outlier = xpp.find_outlier_frames(frames, OUTLIER_THRESHOLD)
print('outlier frames removed: {}'.format(np.flatnonzero(outlier).tolist()))
frames, translations = frames[~outlier], raw['translations'][~outlier]

# 3. One center for the whole scan, then crop about it.
center = xpp.diffraction_center(frames)
print('diffraction center (row, col): {:.2f}, {:.2f}'.format(*center))
frames = xpp.crop_frames(frames, center, FRAME_SIZE)

# 4. The Tukey window multiplies amplitudes; frames are intensities, so its square is used.
frames = frames * xpp.tukey_window(FRAME_SIZE, TUKEY_SHAPE) ** 2

# The recorded translations are (x, y, z) of the sample in meters: rows run along y
# and columns along x.  With the positions as recorded, the object has the orientation
# of the SEM image supplied with the data set.
probe_positions = translations[:, [1, 0]]

# The preprocessing ends here: the scan is complete.
scan = xpt.Scan(frames, probe_positions, wavelength=raw['wavelength'], det_distance=raw['det_distance'],
                det_pixel_pitch=raw['det_pixel_pitch'],
                name='gold balls; data of S. Marchesini et al., CXIDB ID 65, doi:10.11577/1454414')
print(scan.summary())
xpt.save_figures(xpt.view_scan(scan), OUTPUT_DIR)          # one frame and the scan positions

# The model takes the instrument facts and the positions from the scan.
model = xpt.PtychoModel.from_scan(scan, num_probe_modes=NUM_PROBE_MODES)

# The starting probe: one mode computed from the data, with no Fresnel propagation.
init_probe = model.initial_probe(scan)

model.set_params(object_data_fit=OBJECT_DATA_FIT, probe_data_fit=PROBE_DATA_FIT,
                 probe_weight_exponent=PROBE_WEIGHT_EXPONENT, relaxation=RELAXATION,
                 mode_schedule=MODE_SCHEDULE, mode_energy_fraction=MODE_ENERGY_FRACTION,
                 orthogonalize_modes=ORTHOGONALIZE_MODES, initial_probe_distance=MODE_DISTANCE)
model.print_params()

# The most important line: PMACE with no probe given, so the probe is estimated.
recon = model.recon(scan, init_probe=init_probe, iterations=ITERATIONS)

print(recon.summary())
recon.save(OUTPUT_DIR + '/recon.h5')          # the object, the probe, and the record of the run

# View the object, the two probe modes, and the convergence of the data error.
figures = xpt.view_sample(recon, region=VIEW_REGION)
xpt.save_figures(figures, OUTPUT_DIR)

plt.show()          # keep the windows open until they are closed
