.. _PreprocessDocs:

=============
Preprocessing
=============

Preprocessing takes the raw frames of an instrument to a :class:`~xptycho.Scan`.
Each step is a function of arrays in memory that returns a new array, so the
result of every step can be looked at.  Only the reader of a file format
reads a file.

.. code-block:: python

    import xptycho as xpt
    import xptycho.preprocess as xpp

    raw = xpp.cxi.load_raw(path)                                  # counts, dark frames, translations, facts
    frames = xpp.subtract_dark(raw['frames'], raw['dark_frames'])
    outlier = xpp.find_outlier_frames(frames, threshold=2)
    frames, translations = frames[~outlier], raw['translations'][~outlier]
    center = xpp.diffraction_center(frames)
    frames = xpp.crop_frames(frames, center, 512)
    frames = frames * xpp.tukey_window(512, 0.5) ** 2
    scan = xpt.Scan(frames, positions, wavelength=raw['wavelength'],
                    detector_distance=raw['detector_distance'], detector_pitch=raw['detector_pitch'])

Demo 3 runs these steps on the raw gold-ball file.  On that file they
reproduce the preprocessed frames of the reference code.

Steps
-----

.. autofunction:: xptycho.preprocess.subtract_dark

.. autofunction:: xptycho.preprocess.find_outlier_frames

.. autofunction:: xptycho.preprocess.diffraction_center

.. autofunction:: xptycho.preprocess.crop_frames

.. autofunction:: xptycho.preprocess.tukey_window

CXI reader
----------

.. autofunction:: xptycho.preprocess.cxi.load_raw
