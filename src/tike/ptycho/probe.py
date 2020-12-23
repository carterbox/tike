"""Functions related to creating and manipulating probe arrays.

Ptychographic probes are represented as two separate components: a common probe
whose values are the same for all positions and the varying component. The
former is required as it provides the common probe constraint for ptychography
and the later relaxes the former constraint to accomodate real-world
illuminations which may vary with time.

The common components consist of a single array representing multiple
incoherent probes each of which may have an accompanying varying component.

The varying components are stored sparsely as two arrays, so full varying
probes are only reconstructed as needed. The first array is an array of
principal components (coherent modes) that are shared for all positions and the
second is an array of weights for each position.

Each incoherent probe may have its own set of coherent modes. The full varying
probe at a given position is reconstructed by adding the common probe to the
weighted sum of the coherent modes.

```
unique_probe = common_probe + np.sum(weights * coherent_modes)
```

Design comments
---------------
In theory, the probe representation could be implemented in as little as two
arrays: one with all of the shared components and one with the varying
components where the common_probe becomes the first coherent mode. Choosing to
keep the shared coherent_modes separate from the common_probe as a third array
provides backwards compatability and allows for storing fewer coherent_modes in
the case when only some incoherent probes are allowed to vary.

"""

import cupy as cp
import numpy as np

import tike.random


def get_unique(common_probe, m=None, coherent_probe=None, weights=None):
    """Construct the m-th unique probe from a common_probe and weights.

    Parameters
    ----------
    common_probe : (..., 1, 1, INCOH, WIDE, HIGH) complex64
        The common probe amongst all positions.
    m : int or list(int)
        The index of the requested probe
    coherent_probe : (..., 1, COHER, INCOH, WIDE, HIGH) complex64
        The coherent probe for all positions.
    weights : (..., POSI, COHER, INCOH) float32
        The relative intensity of the coherent probes at each position

    Returns
    -------
    unique_probes : (..., POSI, 1, 1, WIDE, HIGH)
    """
    if m is None:
        m = list(range(common_probe.shape[-3]))
    if type(m) is not list:
        m = [m]
    if weights is not None and coherent_probe is not None:
        return common_probe[..., :, m, :, :] + np.sum(
            weights[..., m, None, None] * coherent_probe[..., m, :, :],
            axis=-4,
            keepdims=True,
        )
    else:
        return common_probe[..., :, m, :, :].copy()


def update_coherent_probe(R, coherent_probe, weights, β=0.1):
    """Update coherent probes using residual probe updates.

    This update is copied from the source code of ptychoshelves. It is similar
    to but not the same as equation (31) described by Odstrcil et al (2018). It
    is is also different from updates described in Odstrcil et al (2016).
    However, they aim to correct for probe variation.

    Parameters
    ----------
    R : (..., POSI, 1, 1, WIDE, HIGH) complex64
        Residual probe updates; what's left after subtracting the common probe
        update from the unique probe updates for each position
    coherent_probe : (..., 1, COHER, 1, WIDE, HIGH) complex64
        The coherent probe being updated.
    β : float A relaxation constant that controls how quickly the coherent
        probe modes are updated. Recommended to be < 1 for mini-batch updates.
    weights : (..., POSI) float32
        A vector whose elements are sums of the previous optimal updates for
        each posiiton.

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360

    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iterative
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.
    """
    assert R.shape[-3] == R.shape[-4] == 1
    assert coherent_probe.shape[-3] == 1 == coherent_probe.shape[-5]
    assert R.shape[:-5] == coherent_probe.shape[:-5] == weights.shape[:-1]
    assert weights.shape[-1] == R.shape[-5]
    assert R.shape[-2:] == coherent_probe.shape[-2:]

    # (..., POSI, 1, 1, 1, 1) to match other arrays
    weights = weights[..., None, None, None, None]
    norm_weights = np.linalg.norm(weights, axis=-5, keepdims=True)**2
    if np.all(norm_weights == 0):
        raise ValueError('coherent_probe weights cannot all be zero?')

    # FIXME: What happens when weights is zero!?
    proj = (np.real(R.conj() * coherent_probe) + weights) / norm_weights
    update = np.mean(
        R * np.mean(proj, axis=(-2, -1), keepdims=True),
        axis=-5,
        keepdims=True,
    )
    coherent_probe += β * update / np.linalg.norm(
        update, axis=(-2, -1), keepdims=True)
    assert np.all(np.isfinite(coherent_probe))

    coherent_probe /= np.linalg.norm(coherent_probe,
                                     axis=(-2, -1),
                                     keepdims=True)

    return coherent_probe


def add_modes_random_phase(probe, nmodes):
    """Initialize additional probe modes by phase shifting the first mode.

    Parameters
    ----------
    probe : (:, :, :, M, :, :) array
        A probe with M > 0 incoherent modes.
    nmodes : int
        The number of desired modes.

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    all_modes = np.empty((*probe.shape[:-3], nmodes, *probe.shape[-2:]),
                         dtype='complex64')
    pw = probe.shape[-1]
    for m in range(nmodes):
        if m < probe.shape[-3]:
            # copy existing mode
            all_modes[..., m, :, :] = probe[..., m, :, :]
        else:
            # randomly shift the first mode
            shift = np.exp(-2j * np.pi * (np.random.rand(2, 1) - 0.5) *
                           ((np.arange(0, pw) + 0.5) / pw - 0.5))
            all_modes[..., m, :, :] = (probe[..., 0, :, :] * shift[0][None] *
                                       shift[1][:, None])
    return all_modes


def simulate_variable_weights(scan, coherent_probe):
    """Generate weights for coherent probe that follow random sinusoid.

    The amplitude of the of weights is 1, the phase shift is (0, 2π], and the
    period is at most one full scan.
    """
    N = scan.shape[1]
    x = np.arange(N)[..., :, None, None]
    period = N * np.random.rand(*coherent_probe.shape[:-2])
    phase = 2 * np.pi * np.random.rand(*coherent_probe.shape[:-2])
    return np.sin(2 * np.pi / period * x - phase)


def init_variable_probe(scan, common_probe, N):
    """Initialize arrays for N coherent modes."""

    coherent_probe = tike.random.numpy_complex(
        *common_probe.shape[:-4],
        N,
        *common_probe.shape[-3:],
    ).astype('complex64')
    coherent_probe /= np.linalg.norm(coherent_probe,
                                     axis=(-2, -1),
                                     keepdims=True)

    weights = 1e-6 * np.random.rand(
        *scan.shape[:-1],
        N,
        common_probe.shape[-3],
    ).astype('float32')
    weights -= np.mean(weights, axis=-3, keepdims=True)

    return coherent_probe, weights


def orthogonalize_eig(x):
    """Orthogonalize modes of x using eigenvectors of the pairwise dot product.

    Parameters
    ----------
    x : (..., nmodes, :, :) array_like complex64
        An array of the probe modes vectorized

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    nmodes = x.shape[-3]
    # 'A' holds the dot product of all possible mode pairs. We only fill the
    # lower half of `A` because it is conjugate-symmetric
    A = cp.empty((*x.shape[:-3], nmodes, nmodes), dtype='complex64')
    for i in range(nmodes):
        for j in range(i + 1):
            A[..., i, j] = cp.sum(cp.conj(x[..., i, :, :]) * x[..., j, :, :],
                                  axis=(-1, -2))

    _, vectors = cp.linalg.eigh(A, UPLO='L')
    # np.linalg.eigh guarantees that the eigen values are returned in ascending
    # order, so we just reverse the order of modes to have them sorted in
    # descending order.

    # TODO: Optimize this double-loop
    x_new = cp.zeros_like(x)
    for i in range(nmodes):
        for j in range(nmodes):
            # Sort new modes by eigen value in decending order.
            x_new[..., nmodes - 1 -
                  j, :, :] += vectors[..., i, j, None, None] * x[..., i, :, :]
    assert x_new.shape == x.shape, [x_new.shape, x.shape]

    return x_new


if __name__ == "__main__":
    cp.random.seed(0)
    x = (cp.random.rand(7, 1, 9, 3, 3) +
         1j * cp.random.rand(7, 1, 9, 3, 3)).astype('complex64')
    x1 = orthogonalize_eig(x)
    assert x1.shape == x.shape, x1.shape
