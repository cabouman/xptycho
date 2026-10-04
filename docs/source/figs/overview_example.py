"""A small reconstruction, run when the documentation is built: a known
object is scanned by simulation and reconstructed from the frames alone."""
import numpy as np
import matplotlib.pyplot as plt

import xptycho as xpt

# ------------------------------- Parameters -------------------------------
OBJECT_SIZE = 192
FRAME_SIZE = 32
SCAN_GRID = (17, 17)
SCAN_STEP_PIXELS = 8
MAX_OFFSET_PIXELS = 1.5
PIXEL_PITCH = 1e-8               # m
WAVELENGTH = 1e-10              # m
DET_PIXEL_PITCH = 75e-6          # m
PEAK_PHOTONS = 1e5
ITERATIONS = 100
# --------------------------------------------------------------------------


def make_object(n):
    """Bars, a disk, and a tilted stripe on a smooth background."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    bars = (np.sin(2 * np.pi * 7 * y) > 0.2) & (x > 0.12) & (x < 0.88)
    disk = (x - 0.62) ** 2 + (y - 0.55) ** 2 < 0.16 ** 2
    stripe = np.abs((x - 0.3) + 0.6 * (y - 0.5)) < 0.035
    thickness = 0.5 * bars + 0.8 * disk + 0.6 * stripe + 0.3 * x
    return ((1 - 0.08 * thickness) * np.exp(-0.9j * thickness)).astype(np.complex64)


def make_probe(m):
    """A Gaussian amplitude with a quadratic phase, a focused beam."""
    y, x = np.mgrid[0:m, 0:m] - (m - 1) / 2
    r2 = (x ** 2 + y ** 2) / (0.26 * m) ** 2
    return (np.exp(-r2) * np.exp(1j * 0.8 * r2)).astype(np.complex64)


truth, probe = make_object(OBJECT_SIZE), make_probe(FRAME_SIZE)
probe_positions = xpt.scan_positions(SCAN_GRID, SCAN_STEP_PIXELS * PIXEL_PITCH, MAX_OFFSET_PIXELS * PIXEL_PITCH)
model = xpt.PtychoModel(wavelength=WAVELENGTH, det_pixel_pitch=DET_PIXEL_PITCH, frame_size=FRAME_SIZE,
                        det_distance=PIXEL_PITCH * FRAME_SIZE * DET_PIXEL_PITCH / WAVELENGTH,
                        probe_positions=probe_positions)
model.set_params(object_shape=truth.shape)
model.configure_devices(devices=['cpu'])
scan = model.simulate(xpt.Sample(truth, probe, PIXEL_PITCH), peak_photons=PEAK_PHOTONS)
recon = model.recon(scan, iterations=ITERATIONS, verbose=0)        # no probe given: it is estimated

region = recon.scanned_region()
rows, cols = np.flatnonzero(region.any(axis=1)), np.flatnonzero(region.any(axis=0))
window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
estimate = xpt.match_scale(recon.object, truth, region)
limits = dict(vmin=np.angle(truth[window]).min(), vmax=np.angle(truth[window]).max(), cmap='gray')

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
panels = [(np.log10(scan.frames[len(probe_positions) // 2] + 1), 'one of {} frames\nlog10(counts + 1)'.format(len(probe_positions)), dict(cmap='viridis')),
          (np.angle(estimate[window]), 'reconstructed phase (rad)\nfrom the frames alone', limits),
          (np.angle(truth[window]), 'true phase (rad)', limits),
          (np.abs(recon.probe[0]), 'estimated probe\nmagnitude', dict(cmap='gray'))]
for ax, (image, title, style) in zip(axes, panels):
    shown = ax.imshow(image, **style)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(shown, ax=ax, fraction=0.046)
fig.tight_layout()
