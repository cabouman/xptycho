"""Preprocessing: from the raw frames of an instrument to a
:class:`~xptycho.Scan`.

The general steps are functions of arrays in memory; each returns a new
array and reads or writes no file.  A reader module per file format
(:mod:`xptycho.preprocess.cxi`) does the file reading.
"""
from .utilities import (subtract_dark, find_outlier_frames, diffraction_center, crop_frames,
                        tukey_window)
from . import cxi

__all__ = ['subtract_dark', 'find_outlier_frames', 'diffraction_center', 'crop_frames', 'tukey_window', 'cxi']
