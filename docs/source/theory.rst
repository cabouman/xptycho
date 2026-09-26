.. _Theory:

======
Theory
======

The equations below are those of the PMACE papers
(:ref:`Credits <CreditsDocs>`), with the symbols the papers use.  The
table at the end maps each parameter of the API to its symbol.

The measurement
---------------

The object is a complex transmittance image :math:`x`.  The probe is
represented by :math:`K` mutually incoherent modes :math:`d_k`, each an
:math:`N_p \times N_p` complex array, and :math:`D_k` is the diagonal
matrix of :math:`d_k`.  The operator :math:`P_j` extracts the
:math:`N_p \times N_p` patch of the object under the probe at scan
position :math:`j`, and :math:`v_j = P_j x` is that patch.  With
:math:`F` the orthonormal two-dimensional discrete Fourier transform,
the detector records

.. math::

   y_j = \sqrt{\mathrm{Pois}\Big(\sum_{k=0}^{K-1} |F D_k P_j x|^2\Big)}

where :math:`y_j` is the measured amplitude, the square root of the
counts.  This is the far-field (Fraunhofer) model.

The object update
-----------------

Each scan position is an agent.  Its data-fitting operator replaces the
amplitude of the predicted diffraction pattern by the measured one and
maps the result back to the object:

.. math::

   \tilde v_{j,k} = D_{k,\epsilon}^{-1} F^* \left( y_j \cdot
   \frac{F D_k v_j}{\sqrt{\sum_m |F D_m v_j|^2}} \right), \qquad
   F_j(v_j) = (1 - \alpha_1)\, v_j + \alpha_1 \sum_k w_k\, \tilde v_{j,k}

with mode weights :math:`w_k = \|d_k\|^2 / \sum_m \|d_m\|^2` and the
stable inverse :math:`D_{k,\epsilon}^{-1} = \mathrm{diag}\big(\bar d_k
/ (|d_k|^2 + \epsilon)\big)`.  The parameter :math:`\alpha_1` sets how
far each agent moves toward its data.

The consensus operator averages the patches into one image, weighting
each pixel by the probe's illumination:

.. math::

   \bar x = \sum_k w_k\, \Lambda_k^{-1} \sum_j P_j^T |D_k|^\kappa v_j,
   \qquad
   \Lambda_k = \sum_j P_j^T |D_k|^\kappa P_j, \qquad
   G(v) = [P_0 \bar x, \ldots, P_{J-1} \bar x].

The exponent :math:`\kappa` sets how strongly brightly lit pixels
dominate the average.  The PMACE solution is the state at which every
agent agrees with the consensus, :math:`F(v^*) = G(v^*)`, and it is
found by the Mann iteration

.. math::

   w \leftarrow F(v), \qquad z \leftarrow G(2w - v), \qquad
   v \leftarrow v + 2\rho\,(z - w)

with :math:`\rho` the relaxation parameter.  The reconstructed image
is :math:`\bar x`.

The probe update
----------------

When the probe is estimated, each scan position also holds its own
probe estimate :math:`d_{j,k}`.  Its data-fitting operator is the dual
of the object's, with the patch :math:`X_j = \mathrm{diag}(P_j x)`
playing the role of the probe:

.. math::

   \tilde d_{j,k} = X_{j,\epsilon}^{-1} F^* \left( y_j \cdot
   \frac{F X_j d_{j,k}}{\sqrt{\sum_m |F X_j d_{j,m}|^2}} \right), \qquad
   F_{j,k}(d_{j,k}) = (1 - \alpha_2)\, d_{j,k} + \alpha_2\, \tilde d_{j,k}

and the consensus is the mean over positions,
:math:`\bar d_k = \frac{1}{J} \sum_j d_{j,k}`, with the same Mann
iteration.  A second probe mode is introduced from the residual
intensity, the part of the measurement the current modes do not
explain, and given a chosen fraction of the total probe energy.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - API name
     - symbol
     - meaning
   * - ``object_data_fit``
     - :math:`\alpha_1`
     - weight of the data-fitting step in the object update
   * - ``probe_data_fit``
     - :math:`\alpha_2`
     - weight of the data-fitting step in the probe update
   * - ``probe_weight_exponent``
     - :math:`\kappa`
     - exponent on the probe magnitude in the consensus average
   * - ``relaxation``
     - :math:`\rho`
     - Mann relaxation parameter
   * - ``probe_modes``
     - :math:`K`
     - number of probe modes
   * - ``mode_energy_fraction``
     -
     - fraction of the probe energy given to a newly added mode

The defaults are the values of the 2025 paper: :math:`\rho = 0.5`,
:math:`\kappa = 1.25`, :math:`\alpha_1 = 0.6`, :math:`\alpha_2 = 0.6`.
