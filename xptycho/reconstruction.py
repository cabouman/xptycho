"""The Reconstruction: what a reconstruction returns."""


class Reconstruction:
    """The result of a reconstruction: the complex object image, the probe
    modes, the positions, the convergence curves, and a parameter table
    that records where every value came from.

    The primary result is ``object``, a complex image; its magnitude and
    phase are both retained.  ``phase`` and ``magnitude`` are views for
    display.  The phase is the argument of the transmittance, in
    radians, wrapped to ``(-pi, pi]``, with the sign convention that a
    thicker object gives a more negative phase.

    A ``Reconstruction`` can be passed back to a reconstruction function
    as its starting point, which is how a run is continued.

    Attributes:
        object (ndarray): complex64 ``(rows, cols)``, the transmittance
            image on the object grid.
        pixel_size (float): metres, or None when the scan had no geometry.
        origin (tuple of float): ``(row, col)`` of the grid's first pixel
            in metres, in the coordinates of the scan positions.
        probe (ndarray): complex64 ``(modes, height, width)``, always three
            axes.  The probe the run used or estimated.
        mode_energies (ndarray): ``(modes,)`` fractions summing to one.
        coverage (ndarray): float32 ``(rows, cols)``, the accumulated probe
            weight of the consensus average.  Zero where no probe reached.
            Downstream use should weight by it.
        positions (ndarray): ``(num_frames, 2)`` in metres, the positions
            the run used.
        curves (dict): one list per iteration curve.  ``data_error`` is
            always present: the normalized root mean square error between
            measured and predicted amplitude.  ``object_error`` and
            ``probe_error`` are present when a ground truth was given.
        iterations (int): iterations run, including any continued from.
        extras (dict): algorithm-specific values, labelled by name.
    """

    object = None
    pixel_size = None
    origin = None
    probe = None
    mode_energies = None
    coverage = None
    positions = None
    curves = None
    iterations = 0
    extras = None

    @property
    def phase(self):
        """float32 ``(rows, cols)``: the wrapped phase of ``object`` in
        radians."""
        raise NotImplementedError

    @property
    def magnitude(self):
        """float32 ``(rows, cols)``: the magnitude of ``object``."""
        raise NotImplementedError

    def parameters(self):
        """Return the parameter table.

        Returns:
            list of dict: one row per parameter with keys ``name``,
            ``value``, ``units``, ``origin`` (``given``, ``file``,
            ``derived``, ``default``, or ``estimated``), and ``note``.
            Machine settings (device, batch size, state placement) have
            their own rows, so a reader sees they did not affect the
            answer.
        """
        raise NotImplementedError

    def summary(self):
        """Return the parameter table as text, followed by the iteration
        count, the final errors, the run time, and the device used."""
        raise NotImplementedError

    def show(self, directory=None, compare_to=None):
        """Plot the object magnitude and phase over the coverage, the
        probe modes with their energy fractions, the position shifts if
        any, and the convergence curves.

        Args:
            directory (str, optional): where the figures are saved.  None
                shows them without saving.
            compare_to (GroundTruth, optional): adds the truth and the
                error images, and prints the NRMSE after removing one
                complex scale.
        """
        raise NotImplementedError

    def save(self, directory, compare_to=None):
        """Write one self-contained folder.

        The folder holds ``summary.txt``, ``parameters.csv``, ``recon.h5``
        (object, probe, positions, coverage, curves, with the pixel size
        and provenance as attributes), preview TIFFs of the phase and
        magnitude, and ``plots/``.  :meth:`load` reads it back.

        Args:
            directory (str): the folder to write.
            compare_to (GroundTruth, optional): as in :meth:`show`.
        """
        raise NotImplementedError

    def save_for_laminography(self, directory, tilt_angle=None):
        """Write the projection a laminography reconstruction reads.

        One HDF5 file holding the complex object, the pixel size, the
        origin, the coverage, and the tilt angle, so mbirtorch's
        multi-axis parallel model can consume it directly.

        Args:
            directory (str): the folder to write into.
            tilt_angle (float, optional): the laminography tilt of this
                scan in radians, carried from the scan file when it is
                known.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        """Read a folder written by :meth:`save`.

        Args:
            directory (str): the folder.

        Returns:
            Reconstruction
        """
        raise NotImplementedError
