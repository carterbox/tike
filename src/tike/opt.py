"""Generic implementations of optimization routines.

Generic implementations of optimization algorithm such as conjugate gradient and
line search that can be reused between domain specific modules. In, the future,
this module may be replaced by Operator Discretization Library (ODL) solvers
library.

"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)
randomizer = np.random.default_rng()


def adagrad(d, v=None, eps=1e-6):
    """Return the adaptive gradient algorithm direction.

    Parameters
    ----------
    d : vector
        The current search direction.
    v : vector
        The adagrad gradient weights.
    eps : float
        A tiny constant to prevent zero division.

    Returns
    -------
    d : vector
        The new search direction.
    v : vector
        The new gradient weights.

    """
    if v is None:
        return d, (d * d.conj()).real
    v += (d * d.conj()).real
    d /= (np.sqrt(v) + eps)
    return d, v


def adam(d, v=None, m=None, vdecay=0.9, mdecay=0.999, eps=1e-6):
    """Return the adaptive moment estimation direction.

    Parameters
    ----------
    d : vector
        The current search direction.
    v : vector
        The adam gradient weights.
    m : vector
        The adam momentum weights.
    vdecay, mdecay : float [0, 1)
        A factor which determines how quickly information from previous steps
        decays.
    eps : float
        A tiny constant to prevent zero division.

    Returns
    -------
    d : vector
        The new search direction.
    v : vector
        The new gradient weights.
    m : vector
        The new momentum weights.

    """
    v = 0 if v is None else v
    m = 0 if m is None else m

    m = mdecay * m + (1 - mdecay) * d
    v = vdecay * v + (1 - vdecay) * (d * d.conj()).real

    m_ = m / (1 - mdecay)
    v_ = np.sqrt(v / (1 - vdecay))

    return m_ / (v_ + eps), v, m


def batch_indicies(n, m=1, use_random=False):
    """Return list of indices [0...n) as m groups.

    >>> batch_indicies(10, 3)
    [array([2, 4, 7, 3]), array([1, 8, 9]), array([6, 5, 0])]
    """
    assert 0 < m and m <= n, (m, n)
    i = randomizer.permutation(n) if use_random else np.arange(n)
    return np.array_split(i, m)


def line_search(
    f,
    x,
    d,
    update_multi,
    step_length=1,
    step_shrink=0.5,
):
    """Return a new `step_length` using a backtracking line search.

    Parameters
    ----------
    f : function(x)
        The function being optimized.
    x : vector
        The current position.
    d : vector
        The search direction.
    step_length : float
        The initial step_length.
    step_shrink : float
        Decrease the step_length by this fraction at each iteration.

    Returns
    -------
    step_length : float
        The optimal step length along d.
    cost : float
        The new value of the cost function after stepping along d.

    References
    ----------
    https://en.wikipedia.org/wiki/Backtracking_line_search

    """
    assert step_shrink > 0 and step_shrink < 1
    m = 0  # Some tuning parameter for termination
    fx = f(x)  # Save the result of f(x) instead of computing it many times
    # Decrease the step length while the step increases the cost function
    while True:
        fxsd = f(update_multi(x, step_length, d))
        if fxsd <= fx + step_shrink * m:
            break
        step_length *= step_shrink
        if step_length < 1e-32:
            warnings.warn("Line search failed for conjugate gradient.")
            return 0, fx
    return step_length, fxsd


def direction_dy(xp, grad0, grad1, dir_):
    """Return the Dai-Yuan search direction.

    Parameters
    ----------
    grad0 : array_like
        The gradient from the previous step.
    grad1 : array_like
        The gradient from this step.
    dir_ : array_like
        The previous search direction.

    """
    return (
        - grad1
        + dir_ * xp.linalg.norm(grad1.ravel())**2
        / (xp.sum(dir_.conj() * (grad1 - grad0)) + 1e-32)
    )  # yapf: disable


def update_single(x, step_length, d):
    return x + step_length * d


def dir_single(x):
    return x


def conjugate_gradient(
    array_module,
    x,
    cost_function,
    grad,
    dir_multi=dir_single,
    update_multi=update_single,
    num_iter=1,
    step_length=1,
):
    """Use conjugate gradient to estimate `x`.

    Parameters
    ----------
    array_module : module
        The Python module that will provide array operations.
    x : array_like
        The object to be recovered.
    cost_function : func(x) -> float
        The function being minimized to recover x.
    grad : func(x) -> array_like
        The gradient of cost_function.
    dir_multi : func(x) -> list_of_array
        The dir_ in all GPUs.
    update_multi : func(x) -> list_of_array
        The updated subimages in all GPUs.
    num_iter : int
        The number of steps to take.

    """
    for i in range(num_iter):

        grad1 = grad(x)
        if i == 0:
            dir_ = -grad1
        else:
            dir_ = direction_dy(array_module, grad0, grad1, dir_)
        grad0 = grad1

        dir_list = dir_multi(dir_)

        gamma, cost = line_search(
            f=cost_function,
            x=x,
            d=dir_list,
            update_multi=update_multi,
            step_length=step_length,
        )

        x = update_multi(x, gamma, dir_list)

        logger.debug("step %d; length %.3e -> %.3e; cost %.6e", i, step_length,
                     gamma, cost)

    return x, cost
