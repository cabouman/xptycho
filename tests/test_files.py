"""A scan and a sample are groups of one HDF5 file and read back unchanged."""
import numpy as np

import xptycho as xpt
from tiny import PIXEL_PITCH, make_model, make_object, make_probe


def test_one_file_holds_a_scan_and_a_sample(tmp_path):
    path = str(tmp_path / 'tiny.h5')
    truth = xpt.Sample(make_object(), make_probe(), PIXEL_PITCH, name='tiny')
    model = make_model()
    scan = model.simulate(truth)
    truth.save(path)
    scan.save(path)                       # the same file; the sample is kept

    sample, scan_back = xpt.Sample.load(path), xpt.Scan.load(path)
    assert np.array_equal(sample.object, truth.object) and np.array_equal(sample.probe, truth.probe)
    assert sample.pixel_pitch == PIXEL_PITCH and sample.name == 'tiny' and sample.run is None
    assert np.array_equal(scan_back.frames, scan.frames) and scan_back.detector_pitch == scan.detector_pitch


def test_a_reconstruction_keeps_the_record_of_its_run(tmp_path):
    path = str(tmp_path / 'recon.h5')
    model = make_model()
    probe = make_probe()
    scan = model.simulate(xpt.Sample(make_object(), probe, PIXEL_PITCH))
    recon = model.recon(scan, probe=probe, iterations=3, verbose=0)
    recon.save(path)
    back = xpt.Sample.load(path)
    assert np.array_equal(back.object, recon.object) and back.origin == recon.origin
    assert back.run.iterations == 3 and back.run.data_error == recon.run.data_error
    assert np.array_equal(back.run.coverage, recon.run.coverage)
    assert [row['name'] for row in back.run.parameters] == [row['name'] for row in recon.run.parameters]
    assert np.array_equal(back.scanned_region(), recon.scanned_region())
