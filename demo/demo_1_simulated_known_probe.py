"""Demo 1: reconstruct a simulated scan with a known probe.

This demo follows the synthetic experiment of the PMACE paper (IEEE
Transactions on Computational Imaging, 2023, Figs. 4(d), 5(d), and 6):
a complex object and a 256 x 256 probe, scanned on a 12 x 12 grid at
68 pixel spacing with random offsets, and recorded with Poisson noise
at a peak of 1e4 photons.  The object and the probe are both known, so
the reconstruction is scored against them.  The expected object NRMSE
after 100 iterations is about 0.037.

The ground truth downloads on first run (about 4 MB).  Run from the
repository root.  The run takes a few seconds on a GPU.
"""
import xptycho

# ---------------------------- Parameters ----------------------------
# The instrument.  The detector distance is set below so that the object
# pixel of the model equals the pixel size the ground truth was made at.
ENERGY = 8.8                    # keV
DETECTOR_PIXEL = 75e-6          # m
FRAME_SIZE = 256                # detector pixels per side

# The scan, in units of the object pixel of the ground truth.
SCAN_GRID = (12, 12)            # scan positions along each axis
SCAN_STEP_PIXELS = 68           # object pixels between positions
MAX_OFFSET_PIXELS = 5           # object pixels, random offset of each position
PEAK_PHOTONS = 1e4              # photons at the brightest sample
DARK_RATE = 0.5                 # mean dark counts per pixel
SEED = 0

# The reconstruction.
ITERATIONS = 100
OBJECT_DATA_FIT = 0.7           # alpha_1 in the papers
PROBE_WEIGHT_EXPONENT = 1.5     # kappa
RELAXATION = 0.5                # rho

OUTPUT_DIR = './demo/output/demo_1_simulated_known_probe'
# --------------------------------------------------------------------

# The ground truth: the paper's complex object and probe.
truth = xptycho.demo_truth('synthetic')
pixel = truth.pixel_size

positions = xptycho.scan_positions(SCAN_GRID, SCAN_STEP_PIXELS * pixel, MAX_OFFSET_PIXELS * pixel, seed=SEED)
detector_distance = pixel * FRAME_SIZE * DETECTOR_PIXEL / xptycho.energy_to_wavelength(ENERGY)
model = xptycho.PtychoModel(energy=ENERGY, detector_distance=detector_distance, detector_pixel=DETECTOR_PIXEL,
                            frame_size=FRAME_SIZE, positions=positions)
model.set_params(object_shape=truth.object.shape)

# Simulate the scan: the forward model at every position, then Poisson counts.
scan = model.simulate(truth.object, truth.probe, pixel_size=pixel, peak_photons=PEAK_PHOTONS,
                      dark_rate=DARK_RATE, seed=SEED)
print(scan.summary())
scan.show(OUTPUT_DIR)

model.set_params(object_data_fit=OBJECT_DATA_FIT, probe_weight_exponent=PROBE_WEIGHT_EXPONENT,
                 relaxation=RELAXATION)
model.print_params()

# The most important line: PMACE with the probe known and held fixed.
recon = model.recon(scan, probe=truth.probe, iterations=ITERATIONS)

print(recon.summary())
recon.show(OUTPUT_DIR, compare_to=truth)
recon.save(OUTPUT_DIR)
