"""Demo 2: reconstruct a simulated scan with an unknown two-mode probe.

This demo follows the synthetic experiment of the blind multi-mode PMACE
paper (IEEE Transactions on Computational Imaging, 2025): a complex
object and a probe with two mutually incoherent modes, scanned on a
20 x 20 grid at 36 pixel spacing.  The reconstruction is given only the
data.  It starts from one probe mode computed from the data, adds the
second mode at iteration 20, and estimates the object and both modes.

The ground truth downloads on first run (about 4 MB).  Run from the
repository root.  The run takes about two minutes on a GPU.
"""
import xptycho

# ---------------------------- Parameters ----------------------------
# The instrument.  The detector distance is set below so that the object
# pixel of the model equals the pixel size the ground truth was made at.
WAVELENGTH = 1.4e-9             # m
DETECTOR_PIXEL = 75e-6          # m
FRAME_SIZE = 256                # detector pixels per side

# The scan, in units of the object pixel of the ground truth.
SCAN_GRID = (20, 20)            # scan positions along each axis
SCAN_STEP_PIXELS = 36           # object pixels between positions
MAX_OFFSET_PIXELS = 5           # object pixels, random offset of each position
PEAK_PHOTONS = 1e4              # photons at the brightest sample
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

OUTPUT_DIR = './demo/output/demo_2_simulated_blind_two_modes'
# --------------------------------------------------------------------

# The ground truth: the paper's complex object and its two probe modes.
truth = xptycho.demo_truth('blind')
pixel = truth.pixel_size

positions = xptycho.scan_positions(SCAN_GRID, SCAN_STEP_PIXELS * pixel, MAX_OFFSET_PIXELS * pixel, seed=SEED)
detector_distance = pixel * FRAME_SIZE * DETECTOR_PIXEL / WAVELENGTH
model = xptycho.PtychoModel(wavelength=WAVELENGTH, detector_distance=detector_distance,
                            detector_pixel=DETECTOR_PIXEL, frame_size=FRAME_SIZE, positions=positions,
                            probe_modes=PROBE_MODES)
model.set_params(object_shape=truth.object.shape)

# Simulate the scan from the true object and both true modes.
scan = model.simulate(truth.object, truth.probe, pixel_size=pixel, peak_photons=PEAK_PHOTONS, seed=SEED)
print(scan.summary())
scan.show(OUTPUT_DIR)

model.set_params(object_data_fit=OBJECT_DATA_FIT, probe_data_fit=PROBE_DATA_FIT,
                 probe_weight_exponent=PROBE_WEIGHT_EXPONENT, relaxation=RELAXATION,
                 mode_schedule=MODE_SCHEDULE, mode_energy_fraction=MODE_ENERGY_FRACTION,
                 initial_probe_distance=INITIAL_PROBE_DISTANCE)
model.print_params()

# The most important line: PMACE with no probe given, so the probe is estimated.
recon = model.recon(scan, iterations=ITERATIONS)

print(recon.summary())
recon.show(OUTPUT_DIR, compare_to=truth)
recon.save(OUTPUT_DIR)
