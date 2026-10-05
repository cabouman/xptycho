"""Position refinement finds whole-pixel errors in the positions."""
import numpy as np

import xptycho as xpt
from tiny import PIXEL_PITCH, make_model, make_object, make_probe


def test_refine_positions_recovers_shifts():
    truth, probe = make_object(), make_probe()
    model = make_model()
    scan = model.simulate(xpt.Sample(truth, probe, PIXEL_PITCH))
    true_positions = model.probe_positions.copy()

    wrong = true_positions.copy()
    wrong[10] += (PIXEL_PITCH, 0)          # one pixel down
    wrong[30] += (0, -PIXEL_PITCH)         # one pixel left
    model.set_params(probe_positions=wrong)
    refined, misfit = model.refine_probe_positions(scan, truth, probe, max_shift=1)

    assert np.allclose(refined, true_positions, atol=PIXEL_PITCH / 100)
    assert misfit[10] > 0.5 and misfit[30] > 0.5
    assert misfit[0] == 0
