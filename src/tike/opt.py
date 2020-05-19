"""Generic implementations of optimization routines.

Generic implementations of optimization algorithm such as conjugate gradient and
line search that can be reused between domain specific modules. In, the future,
this module may be replaced by Operator Discretization Library (ODL) solvers
library.

"""

import logging
import warnings

logger = logging.getLogger(__name__)

def line_search_sqr(f, p1, p2, p3, step_length=1, step_shrink=0.5):
        """Optimized line search for square functions
            Example of otimized computation for the Gaussian model:
            sum_j|G_j(psi+gamma dpsi)|^2 = sum_j|G_j(psi)|^2+
                                           gamma^2*sum_j|G_j(dpsi)|^2+
                                           gamma*sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)
            p1 = sum_j|G_j(psi)|^2
            p2 = sum_j|G_j(dpsi)|^2
            p3 = sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)
            Parameters	
            ----------	
            f : function(x)	
                The function being optimized.	
            p1,p2,p3 : vectors	
                Temporarily vectors to avoid computing forward operators        
        """
        
        assert step_shrink > 0 and step_shrink < 1
        m = 0  # Some tuning parameter for termination
        fp1 = f(p1) # optimize computation
        # Decrease the step length while the step increases the cost function
        while f(p1+step_length**2 * p2+step_length*p3) > fp1 + step_shrink * m:
            if step_length < 1e-32:
                warnings.warn("Line search failed for conjugate gradient.")
                return 0
            step_length *= step_shrink            
        return step_length
    
def line_search(f, x, d, step_length=1, step_shrink=0.5, linear=None):
    """Perform a backtracking line search for a partially-linear cost-function.

    For cost functions composed of a non-linear part, f, and a linear part, l,
    such that the cost = f(l(x)), a backtracking line search computations may
    be reduced in exchange for memory because l(x + γ * d) = l(x) + γ * l(d).
    For completely non-linear functions, the linear part is just the identity
    function.

    Parameters
    ----------
    f : function(linear(x))
        The non-linear part of the function being optimized.
    linear : function(x), optional
        The linear part of the function being optimized.
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
    linear = (lambda x: x) if linear is None else linear
    m = 0  # Some tuning parameter for termination
    # Save cache function calls instead of computing them many times
    lx = linear(x)
    ld = linear(d)
    fx = f(lx)
    # Decrease the step length while the step increases the cost function
    while True:
        fxsd = f(lx + step_length * ld)
        if fxsd <= fx + step_shrink * m:
            break
        step_length *= step_shrink
        if step_length < 1e-32:
            warnings.warn("Line search failed for conjugate gradient.")
            return 0, fx
    return step_length, fxsd


def direction_dy(xp, grad0, grad1, dir):
    """Return the Dai-Yuan search direction.

    Parameters
    ----------
    grad0 : array_like
        The gradient from the previous step.
    grad1 : array_like
        The gradient from this step.
    dir : array_like
        The previous search direction.

    """
    return (-grad1 + dir * xp.linalg.norm(grad1.ravel())**2 /
            (xp.sum(dir.conj() * (grad1 - grad0)) + 1e-32))


def conjugate_gradient(
    array_module,
    x,
    cost_function,
    grad,
    num_iter=1,
    linear_function=None,
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
    num_iter : int
        The number of steps to take.

    """
    for i in range(num_iter):
        grad1 = grad(x)
        if i == 0:
            dir = -grad1
        else:
            dir = direction_dy(array_module, grad0, grad1, dir)
        grad0 = grad1
        gamma, cost = line_search(
            f=cost_function,
            linear=linear_function,
            x=x,
            d=dir,
        )
        x = x + gamma * dir
        logger.debug("%4d, %.3e, %.7e", (i + 1), gamma, cost)
    return x, cost
