"""Figures of a scan and of a sample.

Every figure has a heading that says what it is and a caption that says
how to read it.  A figure is saved when a directory is given, and is put
on the screen when the display allows it.
"""
import ast
import os
import textwrap

import numpy as np

from .metrics import match_scale, nrmse

CAPTION_WIDTH = 110        # characters per caption line
# matplotlib backends that write files and cannot open a window
FILE_ONLY_BACKENDS = ('agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template')


def _image(ax, fig, data, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)


def _finish(fig, heading, caption):
    """Add the heading and the caption and leave room for both."""
    lines = textwrap.wrap(caption, CAPTION_WIDTH)
    fig.suptitle(heading, fontsize=13, fontweight='bold')
    fig.text(0.5, 0.012, '\n'.join(lines), ha='center', va='bottom', fontsize=9.5)
    height = fig.get_size_inches()[1]
    fig.tight_layout(rect=[0, (0.22 * len(lines) + 0.15) / height, 1, 1 - 0.25 / height], h_pad=2.0)


def present(figures, directory, block):
    """Save the figures when a directory is given, then show them on the
    screen when the display is interactive.

    Args:
        figures (list): ``(figure, file name)`` pairs.
        directory (str or None): where the figures are saved.
        block (bool): wait until the windows are closed.  False leaves
            them open and returns, so a later ``show`` displays them too.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
        for figure, name in figures:
            figure.savefig(os.path.join(directory, name), dpi=150)
    if matplotlib.get_backend().lower() not in FILE_ONLY_BACKENDS:
        plt.show(block=block)
    else:
        for figure, _ in figures:
            plt.close(figure)


def show_scan(scan, directory=None, block=True):
    """Plot one diffraction frame and the map of scan positions.  See
    :meth:`xptycho.Scan.show`."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    middle = scan.num_frames // 2
    im = axes[0].imshow(np.log10(np.clip(scan.frames[middle], 0, None) + 1), cmap='viridis')
    axes[0].set_title('frame {} of {}, log10(counts + 1)'.format(middle, scan.num_frames))
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].plot(scan.positions[:, 1] * 1e6, scan.positions[:, 0] * 1e6, '.', markersize=3)
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('column (micrometers)')
    axes[1].set_ylabel('row (micrometers)')
    axes[1].set_title('probe center at each scan position')
    heading = 'The scan{}: {} frames of {} x {} pixels'.format(
        ' "{}"'.format(scan.name) if scan.name else '', scan.num_frames, scan.frame_size, scan.frame_size)
    caption = ('Left: one diffraction frame, the far-field intensity the detector recorded at one probe position, '
               'on a log scale.  Right: where the probe was centered on the object for each of the {} frames.  '
               'Neighboring positions overlap, which is what lets the phase be recovered.').format(scan.num_frames)
    _finish(fig, heading, caption)
    present([(fig, 'scan.png')], directory, block)


def _mode_schedule(facts):
    """The iterations at which a mode was added, from the parameter table."""
    value = facts.get('mode_schedule', ())
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = ()
    return [int(i) for i in value]


def show_sample(sample, directory=None, compare_to=None, block=True):
    """Plot the object and the probe modes of a :class:`~xptycho.Sample`,
    and the data error of its run when it has one.  See its ``show``
    method."""
    import matplotlib.pyplot as plt
    recon, record = sample, sample.run
    modes = len(recon.probe)
    plural = '' if modes == 1 else 's'
    if record is None:
        estimated, facts = False, {}
        run = '{} x {} pixels, {} probe mode{}'.format(*recon.object.shape, modes, plural)
    else:
        facts = {row['name']: row['value'] for row in record.parameters}
        estimated = facts.get('probe') == 'estimated'
        run = '{} positions, {} iterations, probe {}, {} mode{}'.format(
            len(record.positions), record.iterations, 'estimated' if estimated else 'known', modes, plural)

    # ------------------------------------------------------------ the object
    covered = np.ones(recon.object.shape, dtype=bool) if record is None else record.coverage > 0
    obj = recon.object
    truth = None
    iterations = None if record is None else record.iterations
    if compare_to is not None:
        truth = np.asarray(compare_to.object)
        covered = recon.scanned_region()
        obj = match_scale(obj, truth, covered)
        error = nrmse(recon.object, truth, covered)
        print('object NRMSE inside the scanned region: {:.6f}'.format(error))

    # Show only the rows and columns that hold something.
    keep_rows, keep_cols = np.flatnonzero(covered.any(axis=1)), np.flatnonzero(covered.any(axis=0))
    window = (slice(keep_rows[0], keep_rows[-1] + 1), slice(keep_cols[0], keep_cols[-1] + 1))
    covered, obj = covered[window], obj[window]

    rows = 2 if truth is not None else 1
    fig, axes = plt.subplots(rows, 2, figsize=(11, 5.5 * rows + 0.9), squeeze=False)
    mag = np.where(covered, np.abs(obj), np.nan)
    phase = np.where(covered, np.angle(obj), np.nan)
    # One gray scale per quantity, shared by the reconstruction and the truth.
    mag_limits = tuple(np.nanpercentile(mag, [1, 99]))
    phase_limits = tuple(np.nanpercentile(phase, [1, 99]))
    _image(axes[0, 0], fig, mag, 'reconstruction, magnitude', 'gray', *mag_limits)
    _image(axes[0, 1], fig, phase, 'reconstruction, phase (radians)', 'gray', *phase_limits)
    if truth is not None:
        truth = truth[window]
        _image(axes[1, 0], fig, np.where(covered, np.abs(truth), np.nan), 'truth, magnitude', 'gray', *mag_limits)
        _image(axes[1, 1], fig, np.where(covered, np.angle(truth), np.nan), 'truth, phase (radians)', 'gray', *phase_limits)
        caption = ('Top row: the reconstructed object after {} iterations, multiplied by the one complex number that '
                   'brings it closest to the truth.  Bottom row: the truth.  Each column uses one gray scale for both '
                   'rows.  Only the scanned region is shown, the rectangle spanned by the probe centers.  The '
                   'normalized root-mean-square error of the object in this region is {:.4f}.').format(
                       iterations, error)
    elif record is None:
        caption = ('The object of the sample: its magnitude (left) and its phase (right).')
    else:
        caption = ('The reconstructed object after {} iterations: its magnitude (left) and its phase (right), '
                   'where the probe reached.  The object is determined up to one complex constant, so only '
                   'differences in phase and ratios of magnitude are meaningful.').format(iterations)
    _finish(fig, ('Object: ' if record is None else 'Reconstructed object: ') + run, caption)

    # ------------------------------------------------------------- the probe
    fig2, axes2 = plt.subplots(modes, 2, figsize=(10, 5 * modes + 0.9), squeeze=False)
    for k in range(modes):
        share = 100 * recon.mode_energies[k]
        _image(axes2[k, 0], fig2, np.abs(recon.probe[k]), 'mode {} magnitude, {:.1f}% of the energy'.format(k, share), 'gray')
        _image(axes2[k, 1], fig2, np.angle(recon.probe[k]), 'mode {} phase (radians)'.format(k), 'twilight')
    if estimated:
        caption = ('The probe {} estimated together with the object, {}.  Left: magnitude, with the share of the '
                   'total probe energy in that mode.  Right: phase.').format(
                       'mode' if modes == 1 else 'modes', 'one row' if modes == 1 else 'one mode per row')
    elif record is None:
        caption = ('The probe of the sample, {}.  Left: magnitude, with the share of the total probe energy in '
                   'that mode.  Right: phase.').format('one row' if modes == 1 else 'one mode per row')
    else:
        caption = ('The probe given to the reconstruction and held fixed, {}.  Left: magnitude.  Right: '
                   'phase.').format('one row' if modes == 1 else 'one mode per row')
    _finish(fig2, ('Probe: ' if record is None else 'Estimated probe: ' if estimated else 'Known probe: ') + run, caption)
    figures = [(fig, 'object.png'), (fig2, 'probe.png')]
    if record is None:
        present(figures, directory, block)
        return

    # ------------------------------------------------------- the convergence
    errors = record.data_error
    fig3, ax3 = plt.subplots(figsize=(8, 5.4))
    ax3.plot(np.arange(1, len(errors) + 1), errors, linewidth=1.8)
    if max(errors) > 10 * min(errors):          # a log axis only when the error spans a decade
        ax3.set_yscale('log')
    ax3.set_xlabel('iteration')
    ax3.set_ylabel('data error')
    ax3.grid(True, which='both', alpha=0.3)
    added = [i for i in _mode_schedule(facts) if 1 <= i <= len(errors)] if estimated else []
    for number, iteration in enumerate(added):
        ax3.axvline(iteration, color='tab:red', linestyle='--', linewidth=1.2,
                    label='a probe mode is added' if number == 0 else None)
    if added:
        ax3.legend()
    caption = ('The data error at each iteration: the root-mean-square difference between the measured amplitudes '
               'and those the forward model predicts from the current object and probe, divided by the '
               'root-mean-square measured amplitude.  It ends at {:.4f}.').format(errors[-1])
    if added:
        caption += '  The dashed line marks the iteration at which a probe mode was added.'
    _finish(fig3, 'Convergence: ' + run, caption)

    present(figures + [(fig3, 'data_error.png')], directory, block)
