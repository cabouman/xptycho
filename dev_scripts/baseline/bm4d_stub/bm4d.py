"""Stand-in for the bm4d package, used only so that pmace.pmace imports.

Every baseline run sets add_reg=False, so neither name is ever called.
"""


class BM4DProfileBM3DComplex:
    pass


def bm4d(*args, **kwargs):
    raise NotImplementedError('bm4d is not available in the baseline environment')
