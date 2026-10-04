"""The tiny problem of baseline B5: a 64 x 64 object and a 16 x 16 probe
from fixed formulas, 49 positions on a 7 x 7 grid at 8 pixel spacing.
The formulas are those of dev_scripts/baseline/capture_tiny.py, which ran
the old ptycho_pmace code on this problem and stored the numbers in
tests/data/tiny_baseline.json."""
import numpy as np

import xptycho as xpt

OBJECT_SIZE, PROBE_SIZE, GRID, SPACING = 64, 16, 7, 8
PIXEL_PITCH, WAVELENGTH, DISTANCE = 1e-8, 1e-10, 1.0


def make_object(n=OBJECT_SIZE):
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    magnitude = 1.0 - 0.15 * (0.5 + 0.5 * np.sin(2 * np.pi * (x + 0.3 * y))) * np.exp(-((x - 0.5) ** 2 + (y - 0.4) ** 2) / 0.12)
    phase = -0.6 * np.exp(-((x - 0.55) ** 2 + (y - 0.5) ** 2) / 0.08) - 0.2 * x
    return (magnitude * np.exp(1j * phase)).astype(np.complex64)


def make_probe(m=PROBE_SIZE):
    y, x = np.mgrid[0:m, 0:m] - (m - 1) / 2
    r2 = (x ** 2 + y ** 2) / (0.32 * m) ** 2
    return (np.exp(-r2) * np.exp(1j * 0.6 * r2)).astype(np.complex64)


def make_positions():
    """Patch centers in meters, row then column."""
    centers = OBJECT_SIZE / 2 + (np.arange(GRID) - (GRID - 1) / 2) * SPACING
    rows, cols = np.meshgrid(np.round(centers), np.round(centers), indexing='ij')
    return np.stack([rows.ravel(), cols.ravel()], axis=1) * PIXEL_PITCH


def make_model(devices=('cpu',), **recon_params):
    """A model of the tiny problem on the 64 x 64 grid of the baseline."""
    model = xpt.PtychoModel(wavelength=WAVELENGTH, det_distance=DISTANCE,
                            det_pixel_pitch=WAVELENGTH * DISTANCE / (PROBE_SIZE * PIXEL_PITCH),
                            frame_size=PROBE_SIZE, probe_positions=make_positions())
    model.set_params(object_shape=(OBJECT_SIZE, OBJECT_SIZE), object_origin=(0.0, 0.0), **recon_params)
    model.configure_devices(devices=list(devices))
    return model


def make_second_mode(m=PROBE_SIZE, scale=0.4):
    """A second probe mode of a different shape, as in
    dev_scripts/baseline/capture_tiny_blind.py."""
    y, x = np.mgrid[0:m, 0:m] - (m - 1) / 2
    return (scale * make_probe(m) * (x / (0.32 * m)) * np.exp(1j * 0.3 * y)).astype(np.complex64)


def numbers(array):
    """Four numbers that pin a complex array."""
    return dict(frobenius=np.linalg.norm(array), sum_real=array.real.sum(), sum_imag=array.imag.sum(),
                max_abs=np.abs(array).max())
