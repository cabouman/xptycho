"""The Reconstruction: what a reconstruction returns."""


class Reconstruction:
    """The result of a reconstruction.

    The primary result is ``object``, the complex transmittance image;
    its magnitude and phase are both retained, and ``phase`` and
    ``magnitude`` are views for display.  The phase is in radians,
    wrapped to ``(-pi, pi]``, with the convention that a thicker object
    gives a more negative phase.  A ``Reconstruction`` can be passed
    back to :meth:`~xptycho.PtychographyModel.recon` as ``init`` to
    continue a run, and :meth:`save` writes the folder a laminography
    reconstruction reads.

    .. list-table::
       :header-rows: 1
       :widths: 18 30 52

       * - attribute
         - type
         - meaning
       * - ``object``
         - complex64 ``(rows, cols)``
         - the transmittance image on the object grid
       * - ``pixel_size``
         - float, metres
         - the object pixel; None when the scan had no geometry
       * - ``origin``
         - ``(row, col)``, metres
         - the grid's first pixel, in the coordinates of the positions
       * - ``probe``
         - complex64 ``(modes, height, width)``
         - the probe the run used or estimated, always three axes
       * - ``mode_energies``
         - float ``(modes,)``
         - the fraction of the probe energy in each mode
       * - ``coverage``
         - float32 ``(rows, cols)``
         - the accumulated probe weight; zero where no probe reached
       * - ``positions``
         - float64 ``(num_frames, 2)``, metres
         - the positions the run used
       * - ``curves``
         - dict of lists
         - ``data_error`` per iteration; ``object_error`` and
           ``probe_error`` when a ground truth was given
       * - ``iterations``
         - int
         - iterations run, including any continued from
       * - ``algorithm``
         - str
         - the algorithm's name
    """

    def __init__(self, object, pixel_size, origin, probe, coverage, positions, curves,
                 iterations, algorithm, parameters):
        raise NotImplementedError

    @property
    def phase(self):
        """float32 ``(rows, cols)``: the wrapped phase of ``object``."""
        raise NotImplementedError

    @property
    def magnitude(self):
        """float32 ``(rows, cols)``: the magnitude of ``object``."""
        raise NotImplementedError

    def parameters(self):
        """Return the parameter table of the run: the model's parameters,
        the algorithm's, and the loop's settings, as rows with keys
        ``name``, ``value``, ``units``, ``origin`` (``given``, ``file``,
        ``derived``, ``default``, or ``estimated``), and ``note``."""
        raise NotImplementedError

    def summary(self):
        """Return the parameter table as text, followed by the iteration
        count, the final errors, the run time, and the devices used."""
        raise NotImplementedError

    def show(self, directory=None, compare_to=None):
        """Plot the object magnitude and phase over the coverage, the
        probe modes with their energy fractions, and the convergence
        curves.

        Args:
            directory (str, optional): where the figures are saved.  None
                shows them without saving.
            compare_to (GroundTruth, optional): adds the truth and the
                error images, and prints the NRMSE after removing one
                complex scale.
        """
        raise NotImplementedError

    def save(self, directory, compare_to=None):
        """Write one self-contained folder: ``summary.txt``,
        ``parameters.csv``, ``recon.h5`` (the object, the probe, the
        positions, the coverage, and the curves, with the pixel size, the
        origin, and the provenance as attributes), preview TIFFs of the
        phase and the magnitude, and ``plots/``.  ``recon.h5`` is the
        file a laminography reconstruction reads.

        Args:
            directory (str): the folder to write.
            compare_to (GroundTruth, optional): as in :meth:`show`.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        """Read a folder written by :meth:`save`."""
        raise NotImplementedError
