"""Demo 1: reconstruct a simulated scan with a known probe.

This demo follows the synthetic experiment of the PMACE paper (IEEE
Transactions on Computational Imaging, 2023, Figs. 4(d), 5(d), and 6):
a complex object and a 256 x 256 probe, scanned on a 12 x 12 grid at
68 pixel spacing with random offsets, and recorded with Poisson noise
at a peak of 1e4 photons.  The object and the probe are both known, so
the reconstruction is scored against them.  The expected object NRMSE
after 100 iterations is about 0.037.

The ground truth downloads on first run (2 MB).  Run from the
repository root.  The run takes a few seconds on a GPU.
"""
import matplotlib.pyplot as plt

import xptycho as xpt

# ---------------------------- Parameters ----------------------------
# The instrument.  The detector distance is set below so that the object
# pixel of the model equals the pixel pitch the ground truth was made at.
ENERGY = 8.8                    # keV
DET_PIXEL_PITCH = 75e-6          # m
FRAME_SIZE = 256                # detector pixels per side

# The scan, in units of the object pixel of the ground truth.
SCAN_GRID = (12, 12)            # scan positions along each axis
SCAN_STEP_PIXELS = 68           # object pixels between positions
MAX_OFFSET_PIXELS = 5           # object pixels, random offset of each position
PEAK_PHOTONS = 1e4              # photons at the brightest detector pixel
DARK_RATE = 0.5                 # mean dark counts per pixel
SEED = 0

# The reconstruction.
ITERATIONS = 100
OBJECT_DATA_FIT = 0.7           # alpha_1 in the papers
PROBE_WEIGHT_EXPONENT = 1.5     # kappa
RELAXATION = 0.5                # rho

# The ground truth, one HDF5 file holding a sample (an object and its probe).
TRUTH_URL = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xptycho_synthetic.h5'
DATA_DIR = './demo/input'
OUTPUT_DIR = './demo/output/demo_1_simulated_known_probe'
# --------------------------------------------------------------------

# Download the ground truth, unless it is already in DATA_DIR, and load it.
# It is a Sample: the 1024 x 1024 complex object and the single 256 x 256
# probe of the 2023 PMACE paper's synthetic experiment.
truth = xpt.Sample.load(xpt.download(TRUTH_URL, DATA_DIR))
pixel_pitch = truth.pixel_pitch

probe_positions = xpt.scan_positions(SCAN_GRID, SCAN_STEP_PIXELS * pixel_pitch, MAX_OFFSET_PIXELS * pixel_pitch, seed=SEED)
det_distance = pixel_pitch * FRAME_SIZE * DET_PIXEL_PITCH / xpt.energy_to_wavelength(ENERGY)
model = xpt.PtychoModel(energy=ENERGY, det_distance=det_distance, det_pixel_pitch=DET_PIXEL_PITCH,
                        frame_size=FRAME_SIZE, probe_positions=probe_positions)
model.set_params(object_shape=truth.object.shape)

# Simulate the scan: the forward model at every position, then Poisson counts.
scan = model.simulate(truth, peak_photons=PEAK_PHOTONS, dark_rate=DARK_RATE, seed=SEED)
print(scan.summary())
xpt.save_figures(xpt.view_scan(scan), OUTPUT_DIR)          # one frame and the scan positions

model.set_params(object_data_fit=OBJECT_DATA_FIT, probe_weight_exponent=PROBE_WEIGHT_EXPONENT,
                 relaxation=RELAXATION)
model.print_params()

# The most important line: PMACE with the probe known and held fixed.
recon = model.recon(scan, probe=truth.probe, iterations=ITERATIONS)

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
