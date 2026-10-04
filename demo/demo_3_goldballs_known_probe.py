"""Demo 3: preprocess and reconstruct measured data, the gold-ball scan.

This demo follows the measured-data experiment of the PMACE paper (IEEE
Transactions on Computational Imaging, 2023).  It starts from the raw file:
800 frames of 621 x 621 detector counts and 20 dark frames.

The data is the Ptychography Gold Ball Example Dataset of Stefano Marchesini
et al., Lawrence Berkeley National Laboratory, collected at beamline 5.3.2.1
of the Advanced Light Source and published in the Coherent X-ray Imaging
Data Bank, cxidb.org, entry 65, doi:10.11577/1454414, under the CC0 public
domain dedication.

The preprocessing is that of the PMACE papers, one step at a time:

1. subtract the mean dark frame;
2. remove the frames whose mean intensity is an outlier;
3. find the center of the diffraction patterns and crop about it;
4. multiply by a Tukey window.

The reconstruction uses a known probe, and its object is shown beside the
one the reference code ptycho_pmace reconstructs.

The raw file (about 630 MB) and the reference (4 MB) download on first run.
Run from the repository root.  The run takes a few minutes on one GPU.
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

# The reference, one HDF5 file holding a sample: the known probe, and the object the
# reference code reconstructs with it.
REFERENCE_URL = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xptycho_goldballs_reference.h5'
DATA_DIR = './demo/input'

# The preprocessing, as in the papers.
OUTLIER_THRESHOLD = 2.0         # standard deviations of the mean frame intensity
FRAME_SIZE = 512                # pixels kept about the diffraction center
TUKEY_SHAPE = 0.5               # fraction of the window inside the taper

# The reconstruction, as in the paper.
ITERATIONS = 100
OBJECT_DATA_FIT = 0.1           # alpha in the paper
PROBE_WEIGHT_EXPONENT = 1.5     # kappa
RELAXATION = 0.5                # rho

# The part of the object to view: first and last row, first and last column, as in the paper.
VIEW_REGION = (100, 500, 100, 500)

OUTPUT_DIR = './demo/output/demo_3_goldballs_known_probe'
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

# The recorded translations are (x, y, z) of the sample in meters.  In the detector
# image of this instrument, rows run along -y and columns along -x.
positions = -translations[:, [1, 0]]

# The preprocessing ends here: the scan is complete.
scan = xpt.Scan(frames, positions, wavelength=raw['wavelength'], detector_distance=raw['detector_distance'],
                detector_pitch=raw['detector_pitch'],
                name='gold balls; data of S. Marchesini et al., CXIDB ID 65, doi:10.11577/1454414')
print(scan.summary())
xpt.save_figures(xpt.view_scan(scan), OUTPUT_DIR)          # one frame and the scan positions

# Load the reference sample: the known probe and the reference code's object.
reference = xpt.Sample.load(xpt.download(REFERENCE_URL, DATA_DIR))

# The model takes the instrument facts and the positions from the scan.  The
# object grid is that of the reference, so the two objects can be compared.
model = xpt.PtychoModel.from_scan(scan)
model.set_params(object_shape=reference.object.shape, object_origin=reference.origin,
                 object_data_fit=OBJECT_DATA_FIT, probe_weight_exponent=PROBE_WEIGHT_EXPONENT,
                 relaxation=RELAXATION)
model.print_params()

# As in the paper, the run starts from an object equal to one everywhere.
start = xpt.Sample(np.ones(reference.object.shape), reference.probe, model.pixel_pitch)

# The most important line: PMACE with the probe given, so it is held fixed.
recon = model.recon(scan, probe=reference.probe, init=start, iterations=ITERATIONS)

print(recon.summary())
recon.save(OUTPUT_DIR + '/recon.h5')          # the object, the probe, and the record of the run

# View the object beside the reference code's, the probe, and the convergence of the data error.
figures = xpt.view_sample(recon, region=VIEW_REGION, compare_to=reference, compare_label='reference code')
xpt.save_figures(figures, OUTPUT_DIR)

plt.show()          # keep the windows open until they are closed
