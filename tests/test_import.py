import xptycho


def test_version():
    assert isinstance(xptycho.__version__, str)
    assert xptycho.__version__
