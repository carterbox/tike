import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search, direction_dy, update_single
from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def combined(
    op,
    pool,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=True, recover_positions=False,
    num_iter=1,
    rtol=-1,
    psi_grad=None,
    psi_dir=None,
    probe_grad=None,
    probe_dir=None,
    **kwargs
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    Parameters
    ----------
    operator : tike.operators.Ptycho
        A ptychography operator.
    pool : tike.pool.ThreadPoolExecutor
        An object which manages communications between GPUs.
    """
    cost0 = np.inf
    for i in range(num_iter):
        if recover_psi:
            psi, cost, psi_grad, psi_dir = update_object(
                op,
                pool,
                data,
                psi,
                scan,
                probe,
                grad0=psi_grad,
                dir0=psi_dir,
            )

        if recover_probe:
            # TODO: add multi-GPU support
            probe, cost, probe_grad, probe_dir = update_probe(
                op,
                pool,
                pool.gather(data, axis=1),
                psi[0],
                pool.gather(scan, axis=1),
                probe[0],
                grad0=probe_grad,
                dir0=probe_dir,
            )
            probe = pool.bcast(probe)

        if recover_positions and pool.num_workers == 1:
            scan, cost = update_positions_pd(
                op,
                pool.gather(data, axis=1),
                psi[0],
                probe[0],
                pool.gather(scan, axis=1),
            )
            scan = pool.bcast(scan)

        # Check for early termination
        if i > 0 and abs((cost - cost0) / cost0) < rtol:
            logger.info("Cost function rtol < %g reached at %d "
                        "iterations.", rtol, i)
            break
        cost0 = cost

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def update_probe(op, pool, data, psi, scan, probe, grad0=None, dir0=None):
    """Solve the probe recovery problem."""

    # TODO: Cache object patche between mode updates
    for m in range(probe.shape[-3]):

        def cost_function(mode):
            return op.cost(data, psi, scan, probe, m, mode)

        def grad(mode):
            # Use the average gradient for all probe positions
            return op.xp.mean(
                op.grad_probe(data, psi, scan, probe, m, mode),
                axis=(1, 2),
                keepdims=True,
            )

        grad1 = grad(probe[..., m:m + 1, :, :])
        dir0 = direction_dy(op.xp, grad0, grad1, dir0)
        grad0 = grad1

        gamma, cost = line_search(
            f=cost_function,
            x=probe[..., m:m + 1, :, :],
            d=dir0,
            update_multi=update_single,
            step_length=4,
        )

        probe[..., m:m + 1, :, :] = update_single(probe[..., m:m + 1, :, :],
                                                  gamma, dir0)

        logger.info('%10s cost is %+12.5e; step %.3e', 'probe', cost, gamma)

    return probe, cost, grad0, dir0


def update_object(op, pool, data, psi, scan, probe, grad0=None, dir0=None):
    """Solve the object recovery problem."""

    def cost_function_multi(psi, **kwargs):
        cost_out = pool.map(op.cost, data, psi, scan, probe)
        # TODO: Implement reduce function for ThreadPool
        cost_cpu = 0
        for c in cost_out:
            cost_cpu += op.asnumpy(c)
        return cost_cpu

    def grad_multi(psi):
        grad_out = pool.map(op.grad, data, psi, scan, probe)
        grad_list = list(grad_out)
        # TODO: Implement reduce function for ThreadPool
        for i in range(1, len(grad_list)):
            grad_cpu_tmp = op.asnumpy(grad_list[i])
            grad_tmp = op.asarray(grad_cpu_tmp)
            grad_list[0] += grad_tmp

        return grad_list[0]

    def update_multi(psi, gamma, dir):

        def f(psi, dir):
            return psi + gamma * dir

        return list(pool.map(f, psi, dir))

    grad1 = grad_multi(psi)
    dir1 = direction_dy(op.xp, grad0, grad1, dir0)
    grad0 = grad1

    dir_list = pool.bcast(dir1)

    gamma, cost = line_search(
        f=cost_function_multi,
        x=psi,
        d=dir_list,
        update_multi=update_multi,
        step_length=8e-5,
    )

    psi = update_multi(psi, gamma, dir_list)

    logger.info('%10s cost is %+12.5e; step %.3e', 'object', cost, gamma)
    return psi, cost, grad0, dir1
