"""The new code against the old ptycho_pmace code on the tiny problem,
with a known probe, on one device and divided among several."""
import json
import os

import numpy as np
import pytest

import xptycho
from tiny import make_model, make_object, make_probe

BASELINE = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'tiny_baseline.json')))
TOLERANCE = 1e-4     # relative; single precision agrees to about 1e-6


def numbers(recon, truth):
    est = xptycho.match_scale(recon.object, truth)
    return dict(frobenius=np.linalg.norm(est), sum_real=est.real.sum(), sum_imag=est.imag.sum(),
                max_abs=np.abs(est).max(), nrmse_obj=xptycho.nrmse(recon.object, truth),
                nrmse_meas=recon.curves['data_error'][-1])


@pytest.mark.parametrize('devices', [('cpu',), ('cpu', 'cpu'), ('cpu',) * 5])
@pytest.mark.parametrize('iterations', [1, 5, 20])
def test_matches_old_code(devices, iterations):
    truth, probe = make_object(), make_probe()
    model = make_model(devices, object_data_fit=BASELINE['obj_data_fit_prm'], relaxation=BASELINE['rho'],
                       probe_weight_exponent=BASELINE['probe_exp'])
    scan = model.simulate(truth, probe)
    recon = model.recon(scan, probe=probe, iterations=iterations, verbose=0)
    mine, old = numbers(recon, truth), BASELINE['iterates'][str(iterations)]
    for name in mine:
        assert mine[name] == pytest.approx(old[name], rel=TOLERANCE), name


def test_devices_agree():
    """Five devices put block boundaries inside scan lines."""
    truth, probe = make_object(), make_probe()
    results = []
    for devices in [('cpu',), ('cpu',) * 5]:
        model = make_model(devices)
        scan = model.simulate(truth, probe)
        results.append(model.recon(scan, probe=probe, iterations=10, verbose=0).object)
    assert np.abs(results[0] - results[1]).max() < 1e-4 * np.abs(results[0]).max()
