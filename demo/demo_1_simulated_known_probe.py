"""Demo 1: reconstruct a simulated scan with a known probe.

This demo follows the synthetic experiment of the PMACE paper (IEEE
Transactions on Computational Imaging, 2023, Figs. 4(d), 5(d), and 6):
an 800 x 800 complex object and a 256 x 256 probe at 8.8 keV, scanned
on a 12 x 12 grid at 68 pixel spacing with random offsets, and recorded
with Poisson noise at a peak of 1e4 photons.  The object and the probe
are both known, so the reconstruction is scored against them.  The
expected object NRMSE after 100 iterations is about 0.037.

The ground truth downloads on first run (about 35 MB).  The run takes
about a minute on a laptop CPU and a few seconds on a GPU.
"""
import xptycho

# ---------------------------- Parameters ----------------------------
# The scan.  The object pixel is 4.52 nm in the paper's simulation; the
# step is 68 of those pixels.  Offsets are uniform on [-5, 5] pixels.
ENERGY = 8.8                    # keV
SCAN_GRID = (12, 12)            # scan positions along each axis
SCAN_STEP = 68 * 4.52e-9        # m between positions
MAX_OFFSET = 5 * 4.52e-9        # m, random offset of each position
PEAK_PHOTONS = 1e4              # peak photons in one frame
DARK_RATE = 0.5                 # mean dark counts per pixel
SEED = 0

# The reconstruction.
ITERATIONS = 100
OBJECT_DATA_FIT = 0.7           # PMACE object weight (alpha_1 in the papers)
PROBE_WEIGHT_EXPONENT = 1.5     # exponent on |probe| in the consensus (kappa)
RELAXATION = 0.5                # Mann step size (rho)

OUTPUT_DIR = './output/demo_1_simulated_known_probe'
# --------------------------------------------------------------------

# The ground truth: the paper's complex object and probe, with the
# object pixel size of the simulation.
truth = xptycho.demo_data('synthetic')

# Simulate the scan: the forward model at every position, then Poisson
# noise.  The result is a Scan holding the frames, the positions, and
# the geometry.
scan = xptycho.simulate_scan(truth, grid=SCAN_GRID, step=SCAN_STEP,
                             max_offset=MAX_OFFSET, energy=ENERGY,
                             peak_photons=PEAK_PHOTONS, dark_rate=DARK_RATE,
                             seed=SEED)
print(scan.summary())
scan.show(OUTPUT_DIR)           # one diffraction frame and the position map

# The forward model: the scan's geometry, the derived object grid, and
# the probe, known and held fixed.
model = xptycho.FarFieldModel.from_scan(scan, probe=truth.probe)
model.print_params()            # every value with its units and where it came from

# The most important line: invert the forward model by PMACE.
recon = model.recon(scan, method='pmace', iterations=ITERATIONS,
                    object_data_fit=OBJECT_DATA_FIT,
                    probe_weight_exponent=PROBE_WEIGHT_EXPONENT,
                    relaxation=RELAXATION)

# Review: the parameter table, then the images and the convergence
# curves, then everything saved to one folder.
print(recon.summary())
recon.show(OUTPUT_DIR, compare_to=truth)
recon.save(OUTPUT_DIR, compare_to=truth)
