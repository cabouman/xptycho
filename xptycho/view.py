"""Figures of a reconstruction."""
import os

import numpy as np

from .metrics import match_scale, nrmse


def _image(ax, fig, data, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)


def show_reconstruction(recon, directory=None, compare_to=None):
    """Plot the object, the probe modes, and the data-error curve of a
    :class:`~xptycho.Reconstruction`.  See its ``show`` method."""
    import matplotlib.pyplot as plt
    facts = {row['name']: row['value'] for row in recon.params}
    run = '{} positions, {} iterations, probe {}, {} mode{}'.format(
        len(recon.positions), recon.iterations, facts.get('probe', ''), len(recon.probe),
        '' if len(recon.probe) == 1 else 's')
    covered = recon.coverage > 0
    obj = recon.object
    truth = None
    if compare_to is not None:
        truth = np.asarray(compare_to.object)
        region = recon.scanned_region()
        obj = match_scale(obj, truth, region)
        covered = region
        print('object NRMSE inside the scanned region: {:.6f}'.format(nrmse(recon.object, truth, region)))

    # Show only the rows and columns that hold something.
    keep_rows, keep_cols = np.flatnonzero(covered.any(axis=1)), np.flatnonzero(covered.any(axis=0))
    window = (slice(keep_rows[0], keep_rows[-1] + 1), slice(keep_cols[0], keep_cols[-1] + 1))
    covered, obj = covered[window], obj[window]
    if truth is not None:
        truth = truth[window]

    rows = 2 if truth is not None else 1
    fig, axes = plt.subplots(rows, 2, figsize=(11, 5 * rows), squeeze=False)
    mag = np.where(covered, np.abs(obj), np.nan)
    phase = np.where(covered, np.angle(obj), np.nan)
    # One gray scale per quantity, shared by the reconstruction and the truth.
    mag_limits = tuple(np.nanpercentile(mag, [1, 99]))
    phase_limits = tuple(np.nanpercentile(phase, [1, 99]))
    _image(axes[0, 0], fig, mag, 'object, magnitude', 'gray', *mag_limits)
    _image(axes[0, 1], fig, phase, 'object, phase (rad)', 'gray', *phase_limits)
    if truth is not None:
        _image(axes[1, 0], fig, np.where(covered, np.abs(truth), np.nan), 'truth, magnitude', 'gray', *mag_limits)
        _image(axes[1, 1], fig, np.where(covered, np.angle(truth), np.nan), 'truth, phase (rad)', 'gray', *phase_limits)
        fig.suptitle('Reconstructed object (top) and truth (bottom), inside the scanned region\n' + run)
    else:
        fig.suptitle('Reconstructed object\n' + run)
    fig.tight_layout()

    modes = len(recon.probe)
    fig2, axes2 = plt.subplots(modes, 2, figsize=(10, 4.5 * modes), squeeze=False)
    for k in range(modes):
        share = 100 * recon.mode_energies[k]
        _image(axes2[k, 0], fig2, np.abs(recon.probe[k]), 'mode {} magnitude, {:.1f}% of energy'.format(k, share), 'gray')
        _image(axes2[k, 1], fig2, np.angle(recon.probe[k]), 'mode {} phase (rad)'.format(k), 'twilight')
    fig2.suptitle('{} probe mode{}, one per row\n'.format(
        'Estimated' if facts.get('probe') == 'estimated' else 'Given', '' if modes == 1 else 's') + run)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.semilogy(np.arange(1, len(recon.curves['data_error']) + 1), recon.curves['data_error'])
    ax3.set_xlabel('iteration')
    ax3.set_ylabel('data error')
    ax3.grid(True, which='both', alpha=0.3)
    ax3.set_title('Data error per iteration\n' + run)
    fig3.tight_layout()

    if directory is None:
        plt.show()
        return
    os.makedirs(directory, exist_ok=True)
    for figure, name in ((fig, 'object.png'), (fig2, 'probe.png'), (fig3, 'data_error.png')):
        figure.savefig(os.path.join(directory, name), dpi=150)
        plt.close(figure)
