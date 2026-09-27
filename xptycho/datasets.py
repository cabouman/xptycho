"""Datasets: the registry, the download helper, and the demo ground truth."""

#: One record per dataset the demos use: url, size in bytes, sha256, and
#: the citation.  Only this registry knows URLs.
DATASETS = {
    'demo_xptycho_data': {
        'url': 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xptycho_data.tgz',
        'size': None, 'sha256': None,
        'citation': 'Ground truth from the PMACE papers (Zhai et al., IEEE TCI 2023, 2025).',
    },
    'goldballs_cxi': {
        'url': 'https://cxidb.org/data/65/AuBalls_700ms_30nmStep_3_full.cxi',
        'size': 632484560, 'sha256': None,
        'citation': ('S. Marchesini, "Ptychography Gold Ball Example Dataset," CXIDB ID 65, '
                     'Lawrence Berkeley National Laboratory, 2017, doi:10.11577/1454414.'),
    },
}


def download_and_extract(url, save_dir=None, sha256=None, expected_size=None):
    """Download a file and, when it is a tar archive, extract it.

    The download resumes if interrupted, is checked against the size and
    checksum when they are given, never asks a question, and prints one
    line per ten percent.  A file already present and passing its check
    is not downloaded again.  When there is no network, the error names
    the URL and the path to put the file at by hand.

    Args:
        url (str): the file.
        save_dir (str, optional): the folder to write into.  Defaults to
            the ``XPTYCHO_DATA_DIR`` environment variable, else
            ``./demo/input``.
        sha256 (str, optional): the expected checksum.
        expected_size (int, optional): the expected size in bytes.

    Returns:
        str: the extracted folder for an archive, else the file path.
    """
    raise NotImplementedError


def fetch(name, save_dir=None):
    """Download a registered dataset and return its local path.

    Args:
        name (str): a key of :data:`DATASETS`.
        save_dir (str, optional): as in :func:`download_and_extract`.

    Returns:
        str: the local path.
    """
    raise NotImplementedError


def demo_data(name, save_dir=None):
    """Return the ground truth of a demo, downloading it on first use.

    Args:
        name (str): ``'synthetic'`` (the 800 x 800 object and 256 x 256
            probe at 8.8 keV of the 2023 paper, object pixel 4.52 nm) or
            ``'blind'`` (the object and two probe modes of the 2025
            paper's two-mode experiment, object pixel 2.4 nm).
        save_dir (str, optional): as in :func:`download_and_extract`.

    Returns:
        GroundTruth
    """
    raise NotImplementedError
