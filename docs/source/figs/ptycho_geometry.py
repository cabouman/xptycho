"""A ptychographic scan: the probe illuminates overlapping patches of the
object, and the detector records the far-field intensity of each."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

OBJECT_FILL = '#f9bcbc'
PANEL_FILL = '#f2e4d4'
PROBE_EDGE = '#c0392b'
RAY_COLOR = '0.45'
FONT_SIZE = 12

fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.set_aspect('equal')
ax.axis('off')

# The source and the focused probe, from the left.
ax.plot(-0.6, 0.0, marker=(8, 1, 0), markersize=14, color='k')
ax.text(-0.6, -0.4, 'Source', fontsize=FONT_SIZE, ha='center', va='top')
for y in (-0.35, 0.35):
    ax.plot([-0.5, 2.6], [y, 0.0], ':', color=RAY_COLOR, lw=1.1)

# The object: a thin plate seen at an angle, with the probe stepping
# across it in overlapping positions.
plate = np.array([[2.2, -1.5], [2.2, 1.5], [3.4, 1.9], [3.4, -1.1]])
ax.add_patch(Polygon(plate, closed=True, facecolor=OBJECT_FILL, edgecolor='k', lw=1.2))
ax.text(2.2, -1.75, 'object', fontsize=FONT_SIZE, va='top')
rng = np.random.default_rng(0)
for u in np.linspace(0.25, 0.75, 3):
    for t in np.linspace(0.18, 0.82, 4):
        p = plate[0] + u * (plate[3] - plate[0]) + t * (plate[1] - plate[0])
        jitter = 0.03 * rng.standard_normal(2)
        ax.add_patch(Circle(p + jitter, 0.3, facecolor='none',
                            edgecolor=PROBE_EDGE, lw=1.3, alpha=0.9))
ax.text(2.8, 2.15, 'overlapping probe positions', fontsize=FONT_SIZE,
        color=PROBE_EDGE, ha='center', va='bottom')

# The far-field cone from one patch to the detector.
patch_center = np.array([2.8, 0.15])
corners = np.array([[6.4, -1.6], [6.4, 1.6], [7.2, 1.9], [7.2, -1.9]])
panel = Polygon(corners, closed=True, facecolor=PANEL_FILL, edgecolor='k', lw=1.4)
ax.add_patch(panel)
for corner in corners:
    ax.plot([patch_center[0], corner[0]], [patch_center[1], corner[1]], ':',
            color=RAY_COLOR, lw=1.1)
ax.text(6.4, -2.05, 'detector: one diffraction\npattern per position',
        fontsize=FONT_SIZE, va='top')

# A schematic diffraction pattern on the panel: concentric rings.
face = np.array([[6.55, -1.35], [6.55, 1.35], [7.05, 1.6], [7.05, -1.6]])
center = face.mean(axis=0)
for r in (0.15, 0.35, 0.55, 0.75, 0.95):
    ring = Circle(center, r, facecolor='none', edgecolor='0.35', lw=1.0, alpha=0.6)
    ax.add_patch(ring)
    ring.set_clip_path(panel)

ax.set_xlim(-1.0, 8.2)
ax.set_ylim(-2.7, 2.7)
