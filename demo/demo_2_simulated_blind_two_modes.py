"""Demo 2: reconstruct a simulated scan with an unknown two-mode probe.

This demo follows the synthetic experiment of the blind multi-mode PMACE
paper (IEEE Transactions on Computational Imaging, 2025): a complex
object and a probe with two mutually incoherent modes, scanned on a
20 x 20 grid at 36 pixel spacing.  The reconstruction is given only the
data.  It starts from one probe mode computed from the data, adds the
second mode at iteration 20, and estimates the object and both modes.

The ground truth downloads on first run (3 MB).  Run from the
repository root.  The run takes about two minutes on a GPU.
"""
import matplotlib.pyplot as plt

import xptycho as xpt

# ---------------------------- Parameters ----------------------------
# The instrument.  The detector distance is set below so that the object
# pixel of the model equals the pixel pitch the ground truth was made at.
WAVELENGTH = 1.4e-9             # m
DETECTOR_PITCH = 75e-6          # m
FRAME_SIZE = 256                # detector pixels per side

# The scan, in units of the object pixel of the ground truth.
SCAN_GRID = (20, 20)            # scan positions along each axis
SCAN_STEP_PIXELS = 36           # object pixels between positions
MAX_OFFSET_PIXELS = 5           # object pixels, random offset of each position
PEAK_PHOTONS = 1e4              # photons at the brightest detector pixel
SEED = 0

# The reconstruction.
PROBE_MODES = 2
ITERATIONS = 200
OBJECT_DATA_FIT = 0.5           # alpha_1 in the papers
PROBE_DATA_FIT = 0.6            # alpha_2
PROBE_WEIGHT_EXPONENT = 1.25    # kappa
RELAXATION = 0.5                # rho
MODE_SCHEDULE = [20]            # the iteration at which the second mode is added
MODE_ENERGY_FRACTION = 0.1      # the share of the probe energy the new mode starts with
INITIAL_PROBE_DISTANCE = 2e-6   # m, Fresnel propagation of the starting probe and of a new mode

# The ground truth, one HDF5 file holding a sample (an object and its probe).
TRUTH_URL = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xptycho_blind.h5'
DATA_DIR = './demo/input'
OUTPUT_DIR = './demo/output/demo_2_simulated_blind_two_modes'
# --------------------------------------------------------------------

# Download the ground truth, unless it is already in DATA_DIR, and load it.
# It is a Sample: the 1078 x 1078 complex object and the two 256 x 256 probe
# modes of the 2025 blind multi-mode PMACE paper.
truth = xpt.Sample.load(xpt.download(TRUTH_URL, DATA_DIR))
pixel_pitch = truth.pixel_pitch

positions = xpt.scan_positions(SCAN_GRID, SCAN_STEP_PIXELS * pixel_pitch, MAX_OFFSET_PIXELS * pixel_pitch, seed=SEED)
detector_distance = pixel_pitch * FRAME_SIZE * DETECTOR_PITCH / WAVELENGTH
model = xpt.PtychoModel(wavelength=WAVELENGTH, detector_distance=detector_distance,
                        detector_pitch=DETECTOR_PITCH, frame_size=FRAME_SIZE, positions=positions,
                        probe_modes=PROBE_MODES)
model.set_params(object_shape=truth.object.shape)

# Simulate the scan from the true object and both true modes.
scan = model.simulate(truth, peak_photons=PEAK_PHOTONS, seed=SEED)
print(scan.summary())
xpt.save_figures(xpt.view_scan(scan), OUTPUT_DIR)          # one frame and the scan positions

model.set_params(object_data_fit=OBJECT_DATA_FIT, probe_data_fit=PROBE_DATA_FIT,
                 probe_weight_exponent=PROBE_WEIGHT_EXPONENT, relaxation=RELAXATION,
                 mode_schedule=MODE_SCHEDULE, mode_energy_fraction=MODE_ENERGY_FRACTION,
                 initial_probe_distance=INITIAL_PROBE_DISTANCE)
model.print_params()

# The most important line: PMACE with no probe given, so the probe is estimated.
recon = model.recon(scan, iterations=ITERATIONS)

print(recon.summary())
recon.save(OUTPUT_DIR + '/recon.h5')          # the object, the probe, and the record of the run

# The error to the truth, inside the rectangle spanned by the probe centers.
region = recon.scanned_region()
print('object NRMSE to the truth: {:.6f}'.format(xpt.nrmse(recon.object, truth.object, region)))

# View the object beside the truth, the probe, and the convergence of the data error.
# With no region given, the view shows the rectangle spanned by the probe centers.
figures = xpt.view_sample(recon, compare_to=truth)
xpt.save_figures(figures, OUTPUT_DIR)

plt.show()          # keep the windows open until they are closed
