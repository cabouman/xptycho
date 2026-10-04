"""The new code against the old ptycho_pmace code on the tiny problem
with two probe modes, the probe estimated and the second mode added at
iteration 2, on one device and divided among several."""
import json
import os

import numpy as np
import pytest

import xptycho as xpt
from tiny import PIXEL_PITCH, make_model, make_object, make_probe, make_second_mode, numbers

BASELINE = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'tiny_blind_baseline.json')))
TOLERANCE = 1e-3     # relative; the first iterations agree to about 1e-5


def setup(devices):
    truth = make_object()
    modes = np.stack([make_probe(), make_second_mode(scale=BASELINE['second_mode_scale'])])
    model = make_model(devices, object_data_fit=BASELINE['obj_data_fit_prm'],
                       probe_data_fit=BASELINE['probe_data_fit_prm'], relaxation=BASELINE['rho'],
                       probe_weight_exponent=BASELINE['probe_exp'], mode_schedule=BASELINE['add_mode'],
                       mode_energy_fraction=BASELINE['energy_ratio'],
                       initial_probe_distance=BASELINE['propagation_dist'])
    model.probe_modes = 2
    return model, model.simulate(xpt.Sample(truth, modes, PIXEL_PITCH)), truth


def close(mine, old):
    for name in old:
        assert mine[name] == pytest.approx(old[name], rel=TOLERANCE, abs=1e-4), name


def test_start_matches_old_code():
    model, scan, _ = setup(('cpu',))
    probe = model.initial_probe(scan)
    close(numbers(probe[0]), BASELINE['init_probe'])
    close(numbers(model.initial_object(scan, probe)), BASELINE['init_obj'])


@pytest.mark.parametrize('devices', [('cpu',), ('cpu',) * 3])
@pytest.mark.parametrize('iterations', [1, 3, 6])
def test_matches_old_code(devices, iterations):
    model, scan, truth = setup(devices)
    recon = model.recon(scan, iterations=iterations, verbose=0)
    old = BASELINE['iterates'][str(iterations)]
    close(numbers(xpt.match_scale(recon.object, truth)), old['object'])
    assert len(recon.probe) == len(old['probe'])
    for mine, theirs in zip(recon.probe, old['probe']):
        close(numbers(mine), theirs)
    assert recon.run.data_error[-1] == pytest.approx(old['nrmse_meas'], rel=TOLERANCE)
