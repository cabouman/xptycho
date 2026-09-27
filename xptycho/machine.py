"""Machine settings: where the arithmetic runs and where the state lives."""


class Machine:
    """The settings that decide where a reconstruction runs, none of
    which changes the answer.

    A reconstruction chooses all of these itself and prints its choices.
    Pass a ``Machine`` only to override them, for example to force the
    CPU, to shrink the batch size on a small GPU, or to keep the
    per-position state in a file when host memory is short.

    The result of a PMACE reconstruction does not depend on the batch
    size: nothing computed inside one pass over the frames is used
    inside the same pass, so the consensus sums are the same however
    the frames are split.  The test suite checks this.

    Args:
        device (str, optional): ``'cpu'``, ``'cuda'``, ``'cuda:1'``, or a
            list of CUDA devices on one node, across which the scan
            positions are split.  Defaults to every CUDA device present,
            else the CPU.
        batch_size (int, optional): frames per batch.  Defaults to the
            largest batch that fits the device with room for the object
            and the accumulators.  Fixed within a run, so one FFT plan
            serves every batch.
        state (str, optional): where the per-position state of the
            iteration lives: ``'device'``, ``'host'``, or ``'file'``.
            Defaults to the first that fits.  The state is the largest
            array of the problem, eight bytes per measured sample.  A
            file lives in the output folder and is deleted when the run
            ends.
        probe_state_positions (int, optional): how many scan positions
            carry a per-position probe estimate in a blind
            reconstruction.  Defaults to all positions up to 4096, chosen
            spatially uniformly.  The object update uses only the
            average probe, so this bounds memory without changing the
            object update.
        deterministic (bool, optional): use a deterministic reduction for
            the consensus sums instead of atomic adds.  Slower; for
            tests.  Defaults to False.
    """

    def __init__(self, device=None, batch_size=None, state=None,
                 probe_state_positions=None, deterministic=False):
        raise NotImplementedError

    def summary(self):
        """Return the settings as text, each marked chosen or given."""
        raise NotImplementedError
