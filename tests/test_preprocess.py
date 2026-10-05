"""The preprocessing steps on small arrays with known answers."""
import numpy as np

import xptycho.preprocess as xpp


def test_subtract_dark_clips_at_zero():
    frames = np.array([[[5.0, 1.0]], [[3.0, 9.0]]])
    dark = np.array([[[2.0, 2.0]], [[4.0, 4.0]]])           # mean dark is 3
    assert np.array_equal(xpp.subtract_dark(frames, dark), [[[2.0, 0.0]], [[0.0, 6.0]]])


def test_find_outlier_frames():
    frames = np.ones((20, 4, 4))
    frames[7] *= 3
    outlier = xpp.find_outlier_frames(frames, threshold=2)
    assert outlier.tolist() == [j == 7 for j in range(20)]


def test_center_and_crop_put_the_pattern_at_the_frame_center():
    frames = np.zeros((3, 21, 21))
    frames[:, 12, 8] = 1.0                                   # every pattern at row 12, column 8
    center = xpp.diffraction_center(frames)
    assert center == (12.0, 8.0)
    cropped = xpp.crop_frames(frames, center, 8)
    assert cropped.shape == (3, 8, 8) and np.all(cropped[:, 4, 4] == 1) and cropped.sum() == 3


def test_tukey_window():
    window = xpp.tukey_window(64, 0.5)
    assert window.shape == (64, 64) and window[32, 32] == 1 and window[0, 0] == 0
    assert np.allclose(window, window.T) and window.min() >= 0 and window.max() == 1
