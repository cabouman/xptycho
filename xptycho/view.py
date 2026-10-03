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
    covered = recon.coverage > 0
    obj = recon.object
    truth = None
    if compare_to is not None:
        truth = np.asarray(compare_to.object)
        obj = match_scale(obj, truth, recon.coverage)
        print('object NRMSE over the covered pixels: {:.6f}'.format(nrmse(recon.object, truth, recon.coverage)))

    rows = 2 if truth is not None else 1
    fig, axes = plt.subplots(rows, 2, figsize=(11, 5 * rows), squeeze=False)
    mag = np.where(covered, np.abs(obj), np.nan)
    phase = np.where(covered, np.angle(obj), np.nan)
    limits = {}
    if truth is not None:
        t_mag = np.where(covered, np.abs(truth), np.nan)
        t_phase = np.where(covered, np.angle(truth), np.nan)
        limits = dict(mag=(np.nanmin(t_mag), np.nanmax(t_mag)), phase=(np.nanmin(t_phase), np.nanmax(t_phase)))
        _image(axes[1, 0], fig, t_mag, 'truth, magnitude', 'gray', *limits['mag'])
        _image(axes[1, 1], fig, t_phase, 'truth, phase (rad)', 'twilight', *limits['phase'])
    _image(axes[0, 0], fig, mag, 'object, magnitude', 'gray', *limits.get('mag', (None, None)))
    _image(axes[0, 1], fig, phase, 'object, phase (rad)', 'twilight', *limits.get('phase', (None, None)))
    fig.tight_layout()

    modes = len(recon.probe)
    fig2, axes2 = plt.subplots(modes, 2, figsize=(8, 4 * modes), squeeze=False)
    for k in range(modes):
        share = 100 * recon.mode_energies[k]
        _image(axes2[k, 0], fig2, np.abs(recon.probe[k]), 'probe mode {}, magnitude ({:.1f}% of energy)'.format(k, share), 'gray')
        _image(axes2[k, 1], fig2, np.angle(recon.probe[k]), 'probe mode {}, phase (rad)'.format(k), 'twilight')
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.semilogy(np.arange(1, len(recon.curves['data_error']) + 1), recon.curves['data_error'])
    ax3.set_xlabel('iteration')
    ax3.set_ylabel('data error')
    ax3.grid(True, which='both', alpha=0.3)
    fig3.tight_layout()

    if directory is None:
        plt.show()
        return
    os.makedirs(directory, exist_ok=True)
    for figure, name in ((fig, 'object.png'), (fig2, 'probe.png'), (fig3, 'data_error.png')):
        figure.savefig(os.path.join(directory, name), dpi=150)
        plt.close(figure)
