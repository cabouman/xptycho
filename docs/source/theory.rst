.. _Theory:

======
Theory
======

This page describes the PMACE algorithm of the two TCI papers
(:ref:`Credits <CreditsDocs>`): the 2023 paper, where the probe is
known, and the 2025 paper, where the probe is estimated and may have
several modes.  The 2023 algorithm is the special case :math:`K = 1`
with the probe held fixed, so this page describes one algorithm.  The
symbols are those of the papers; the table at the end maps each
parameter of the API to its symbol.

The quantities
--------------

The algorithm uses four kinds of quantities.  They differ in who sets
them and how long they live.

Measured data
   Fixed for the whole run and never modified.

   - :math:`y_j`, one per scan position :math:`j = 0, \ldots, J-1`: the
     measured amplitude, an :math:`N_p \times N_p` real array.  It is
     the square root of the detector counts after dark subtraction and
     masking.
   - The scan positions.  They define :math:`P_j`, the operator that
     cuts patch :math:`j` out of the object.
   - Instrument facts (wavelength, distance, detector pixel).  The
     algorithm never uses them.  They convert positions from meters to
     pixels and give the pixel size of the answer.

Unknowns
   Estimated by the run.

   - :math:`x`: the object, an :math:`N_1 \times N_2` complex
     transmittance image.  The output.
   - :math:`d_k`, :math:`k = 0, \ldots, K-1`: the probe modes, each an
     :math:`N_p \times N_p` complex array.  Known and fixed in the 2023
     paper; estimated in the 2025 paper.

Design parameters
   Chosen by the user and never estimated.

   - :math:`\alpha_1`, the object data-fit weight, and
     :math:`\alpha_2`, the probe data-fit weight.
   - :math:`\kappa \in [1, 2]`, the exponent on :math:`|d_k|` in the
     averaging weights.
   - :math:`\rho \in (0, 1)`, the Mann step.
   - :math:`K`, the number of modes, the iteration at which each new
     mode is added, and the energy fraction a new mode gets.
   - The number of iterations.

   The :math:`\epsilon` of the stable inverse is a fixed formula,
   :math:`10^{-6} \sqrt{\|d\|^2 / \dim d}`, not a choice.

Run state
   Exists only during the run and is discarded after.

   - :math:`\mathbf{v} = [v_0, \ldots, v_{J-1}]`: one complex
     :math:`N_p \times N_p` patch per scan position.  This is the PMACE
     state.  It is the same size as the data.
   - When the probe is estimated:
     :math:`\mathbf{s}_k = [d_{0,k}, \ldots, d_{J-1,k}]`, one probe
     estimate per position, per mode.  Also the size of the data, once
     per mode.
   - Temporaries :math:`\mathbf{w}, \mathbf{z}` (patches) and
     :math:`\mathbf{r}_k, \mathbf{u}_k` (probes).

   The object :math:`x` is not state.  It is computed from
   :math:`\mathbf{v}` by the weighted average whenever it is needed,
   and that same average is the final answer.

The operators
-------------

Building blocks.  :math:`P_j` cuts out patch :math:`j`, and
:math:`P_j^T` adds a patch back into the image at position :math:`j`.
:math:`\mathcal{F}` and :math:`\mathcal{F}^*` are the orthonormal
two-dimensional discrete Fourier transform and its inverse; this is
the far-field (Fraunhofer) model, under which the detector records

.. math::

   y_j = \sqrt{\mathrm{Pois}\Big(\sum_{k=0}^{K-1} |\mathcal{F} D_k P_j x|^2\Big)} .

:math:`D_k = \mathrm{diag}(d_k)`, with the stable inverse
:math:`D_{k,\epsilon}^{-1} = \mathrm{diag}\big(d_k^* / (|d_k|^2 +
\epsilon)\big)`.  :math:`\Lambda_k = \sum_j P_j^T |D_k|^\kappa P_j` is
the coverage map, an image-sized diagonal; it changes whenever the
probe changes.  :math:`w_k = \|d_k\|^2 / \sum_m \|d_m\|^2` is the
energy weight of mode :math:`k`.

**Patch data-fit step** :math:`F^I_j`: one patch in, one patch out.
It uses only :math:`y_j`, :math:`v_j`, and the modes.

.. math::

   \tilde v_{j,k} = D_{k,\epsilon}^{-1}\, \mathcal{F}^* \left( y_j \circ
   \frac{\mathcal{F} D_k v_j}{\sqrt{\sum_m |\mathcal{F} D_m v_j|^2}} \right), \qquad
   F^I_j(v_j) = (1 - \alpha_1)\, v_j + \alpha_1 \sum_k w_k\, \tilde v_{j,k}

**Patch averaging step** :math:`\mathbf{G}^I`: all patches in, all
patches out, through the image.

.. math::

   \bar x = \sum_k w_k\, \Lambda_k^{-1} \sum_j P_j^T |D_k|^\kappa v_j,
   \qquad (\mathbf{G}^I \mathbf{v})_j = P_j\, \bar x

**Probe data-fit step** :math:`F^P_{j,k}`: uses the consensus patch
:math:`z_j`, :math:`y_j`, and all modes at position :math:`j`.

.. math::

   \tilde d_{j,k} = X_{j,\epsilon}^{-1}\, \mathcal{F}^* \left( y_j \circ
   \frac{\mathcal{F} X_j d_{j,k}}{\sqrt{\sum_m |\mathcal{F} X_j d_{j,m}|^2}} \right),
   \quad X_j = \mathrm{diag}(z_j), \qquad
   F^P_{j,k}(d_{j,k}) = (1 - \alpha_2)\, d_{j,k} + \alpha_2\, \tilde d_{j,k}

**Probe averaging step** :math:`\mathbf{G}^P`: the plain mean over
positions, :math:`\bar d_k = \frac{1}{J} \sum_j d_{j,k}`.

One iteration
-------------

Algorithm 1 of the 2025 paper.  A line is *local* when every position
is handled on its own, and *global* when it needs a sum over all
positions.

.. list-table::
   :header-rows: 1
   :widths: 10 36 54

   * -
     - step
     - what it does
   * - local
     - :math:`\mathbf{w} \leftarrow \mathbf{F}^I(\mathbf{v})`
     - each position fits its own data
   * - global
     - :math:`\mathbf{z} \leftarrow \mathbf{G}^I(2\mathbf{w} - \mathbf{v})`
     - one scatter-add over all :math:`j`, divide by :math:`\Lambda`, one gather
   * - local
     - :math:`\mathbf{v} \leftarrow \mathbf{v} + 2\rho\,(\mathbf{z} - \mathbf{w})`
     - Mann update
   * -
     - for each mode :math:`k`:
     - when the probe is estimated
   * - local
     - :math:`\quad \mathbf{r}_k \leftarrow \mathbf{F}^P_k(\mathbf{s}_k;\, \mathbf{z})`
     - each position refines its own probe, using its consensus patch :math:`z_j`
   * - global
     - :math:`\quad \mathbf{u}_k \leftarrow \mathbf{G}^P(2\mathbf{r}_k - \mathbf{s}_k)`
     - one mean over all :math:`j`
   * - local
     - :math:`\quad \mathbf{s}_k \leftarrow \mathbf{s}_k + 2\rho\,(\mathbf{u}_k - \mathbf{r}_k)`
     - Mann update
   * -
     - add a mode if scheduled
     - from the residual intensity; all modes rescaled so the total energy is unchanged
   * -
     - output :math:`\hat x = \bar x(\mathbf{v})`, :math:`\hat d_k = \bar d_k(\mathbf{s}_k)`
     - the weighted average of the patches; the mean of the probe estimates

Every pass has the same shape: a local step, in which each position
uses its own :math:`y_j`, its own state row, and the current consensus;
a global step, one sum over all positions, broadcast back; and a local
step, in which each position updates its own state row.  This happens
once per iteration for the patches, and once more per mode for the
probes when the probe is estimated.  That is the entire communication
pattern of the algorithm.

Initialization
--------------

The probe first, from the mean back-transformed data, propagated a
chosen distance :math:`z` by Fresnel propagation :math:`U`:

.. math::

   d^{(0)} = U_{\eta, z, \Delta_x} \left\{ \frac{1}{J} \sum_j
   (P_j \mathbf{1})^{-1}\, \mathcal{F}^* y_j \right\}

Then the object, so that each patch has the right scale:

.. math::

   x^{(0)} = \Lambda_0^{-1} \sum_j P_j^T \left( \frac{\|y_j\|}{\|d^{(0)}\|}\,
   \mathbf{1} \right), \qquad \Lambda_0 = \sum_j P_j^T P_j

A new mode comes from the residual intensity the current modes do not
explain, :math:`I_j - \sum_k |\mathcal{F} D_k x_j|^2`, back-transformed,
divided by the patch, averaged over positions, and Fresnel propagated.
All modes are then rescaled so the total energy is unchanged.

What this means for the implementation
--------------------------------------

1. **Every pass is local, global, local.**  The local steps need only
   their own :math:`y_j`, their own state row, and the current
   consensus.  The global step is one sum over :math:`j`.  This is what
   lets the data stream through in batches and across GPUs without
   changing the answer.
2. **The state is as large as the data, times** :math:`(1 + K)`.  For
   a scan that does not fit in memory, where :math:`\mathbf{v}` and
   :math:`\mathbf{s}_k` live is the first design question.
3. **The averaging step is scatter-add, divide by coverage, gather.**
   Nothing else touches the whole image.  :math:`\Lambda_k` is itself a
   scatter-add of :math:`|d_k|^\kappa`, recomputed when the probe
   changes.
4. **The probe step uses the consensus patch** :math:`z_j`, not
   :math:`v_j`.  The probe pass needs the patches from the image pass
   of the same iteration, either kept per position or regathered from
   :math:`\bar x`.
5. **The physics is in three places only:** :math:`P_j` (the
   positions), :math:`\mathcal{F}` (far-field propagation), and
   :math:`D_k` (the probe).  Everything else is arithmetic on patches.

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
