import logging

import cupy as cp

from tike.linalg import lstsq, projection

from ..position import update_positions_pd
from ..probe import orthogonalize_eig, get_unique, update_coherent_probe

logger = logging.getLogger(__name__)


def lstsq_grad(
    op, pool,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=False, recover_positions=False,
    cg_iter=4,
    cost=None,
    coherent_probe=None,
    weights=None,
):  # yapf: disable
    """Solve the ptychography problem using Odstrcil et al's approach.

    The near- and farfield- ptychography problems are solved separately using
    gradient descent in the farfield and linear-least-squares in the nearfield.

    Parameters
    ----------
    op : tike.operators.Ptycho
        A ptychography operator.
    pool : tike.pool.ThreadPoolExecutor
        An object which manages communications between GPUs.

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iterative
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.

    """
    xp = op.xp
    data_ = data[0]
    probe = probe[0]
    scan_ = scan[0]
    psi = psi[0]
    if coherent_probe is not None:
        weights = weights[0]
        coherent_probe = coherent_probe[0]

    common_probe = probe
    unique_probe = get_unique(probe,
                              coherent_probe=coherent_probe,
                              weights=weights)

    # Compute the diffraction patterns for all of the probe modes at once.
    # We need access to all of the modes of a position to solve the phase
    # problem. The Ptycho operator doesn't do this natively, so it's messy.
    patches = cp.zeros(data_.shape, dtype='complex64')
    patches = op.diffraction._patch(
        patches=patches,
        psi=psi,
        scan=scan_,
        fwd=True,
    )
    patches = patches.reshape(op.ntheta, scan_.shape[-2], 1, 1,
                              op.detector_shape, op.detector_shape)

    nearplane = op.xp.tile(patches, reps=(1, 1, 1, probe.shape[-3], 1, 1))
    pad, end = op.diffraction.pad, op.diffraction.end
    nearplane[..., pad:end, pad:end] *= unique_probe

    # Solve the farplane phase problem
    farplane = op.propagation.fwd(nearplane, overwrite=False)
    intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
    cost = op.propagation.cost(data_, intensity)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)
    farplane -= 0.5 * op.propagation.grad(data_, farplane, intensity)

    if __debug__:
        intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
        cost = op.propagation.cost(data_, intensity)
        logger.info('%10s cost is %+12.5e', 'farplane', cost)
        # TODO: Only compute cost every 20 iterations or on a log sampling?

    # Use χ (chi) to solve the nearplane problem. We use least-squares to
    # find the update of all the search directions: object, probe,
    # positions, etc that causes the nearplane wavefront to match the one
    # we just found by solving the phase problem.
    farplane = op.propagation.adj(farplane, overwrite=True)
    chi = [
        farplane[..., m:m + 1, :, :] - nearplane[..., m:m + 1, :, :]
        for m in range(probe.shape[-3])
    ]

    # To solve the least-squares optimal step problem we flatten the last
    # two dimensions of the nearplanes and convert from complex to float
    lstsq_shape = (*nearplane.shape[:-3], 1,
                   nearplane.shape[-2] * nearplane.shape[-1] * 2)

    for m in range(probe.shape[-3]):
        chi_ = chi[m]
        probe_ = common_probe[..., m:m + 1, :, :]
        uprobe_ = unique_probe[..., m:m + 1, :, :]

        logger.info('%10s cost is %+12.5e', 'nearplane',
                    cp.linalg.norm(cp.ravel(chi_)))

        updates = []

        if recover_psi:
            # FIXME: Implement conjugate gradient
            grad_psi = chi_.copy()
            grad_psi[..., pad:end, pad:end] *= cp.conj(uprobe_)

            probe_intensity = cp.ones(
                (*scan_.shape[:2], 1, 1, 1, 1),
                dtype='complex64',
            ) * cp.square(cp.abs(uprobe_))

            norm_probe = op.diffraction._patch(
                patches=probe_intensity,
                psi=cp.ones_like(psi),
                scan=scan_,
                fwd=False,
            ) + 1e-6

            # FIXME: What to do when elements of this norm are zero?
            dir_psi = op.diffraction._patch(
                patches=grad_psi,
                psi=cp.zeros_like(psi),
                scan=scan_,
                fwd=False,
            ) / norm_probe

            dOP = op.diffraction._patch(
                patches=cp.zeros((*scan_.shape[:2], *data_.shape[-2:]),
                                 dtype='complex64'),
                psi=dir_psi,
                scan=scan_,
                fwd=True,
            )
            dOP = dOP.reshape(op.ntheta, scan_.shape[-2], 1, 1,
                              op.detector_shape, op.detector_shape)
            dOP[..., pad:end, pad:end] *= uprobe_

            updates.append(dOP)

        if recover_probe:
            patches = op.diffraction._patch(
                patches=cp.zeros(data_.shape, dtype='complex64'),
                psi=psi,
                scan=scan_,
                fwd=True,
            )
            patches = patches.reshape(op.ntheta, scan_.shape[-2], 1, 1,
                                      op.detector_shape, op.detector_shape)

            # (24a) steepest descent update
            grad_probe = (chi_ * xp.conj(patches))[..., pad:end, pad:end]

            psi_intensity = cp.square(cp.abs(patches[..., pad:end, pad:end]))
            norm_psi = cp.sum(psi_intensity, axis=1, keepdims=True) + 1e-6

            # (25a) common update direction
            dir_probe = cp.sum(grad_probe, axis=1, keepdims=True) / norm_psi

            # ΔPO from (21)
            dPO = patches.copy()
            dPO[..., pad:end, pad:end] *= dir_probe

            updates.append(dPO)

        if recover_probe and coherent_probe is not None:
            logger.info('Updating coherent probes')
            # (30) residual probe updates
            R = grad_probe - cp.mean(grad_probe, axis=-5, keepdims=True)

            for c in range(coherent_probe.shape[-4]):

                coherent_probe[
                    ..., c:c + 1, m:m + 1, :, :] = update_coherent_probe(
                        R,
                        coherent_probe[..., c:c + 1, m:m + 1, :, :],
                        weights[..., c, m],
                        β=0.01,  # TODO: Adjust according to mini-batch size
                    )

                # Determine new weights for the updated coherent probe
                phi = patches.copy()
                phi[..., pad:end, pad:end] *= coherent_probe[..., c:c + 1,
                                                             m:m + 1, :, :]
                n = cp.mean(
                    cp.real(chi_ * phi.conj()),
                    axis=(-1, -2),
                    keepdims=True,
                )
                norm_phi = cp.square(cp.abs(phi))
                d = cp.mean(norm_phi, axis=(-1, -2), keepdims=True)
                d += 0.1 * cp.mean(d, axis=-5, keepdims=True)
                weight_update = (n / d).reshape(*weights[..., 0, 0].shape)
                assert cp.all(cp.isfinite(weight_update))

                # (33) The sum of all previous steps constrained to zero-mean
                weights[..., c, m] += weight_update
                weights[..., c, m] -= cp.mean(
                    weights[..., c, m],
                    axis=-1,
                    keepdims=True,
                )

                if coherent_probe.shape[-4] <= c + 1:
                    # Subtract projection of R onto new probe from R
                    R -= projection(
                        R,
                        coherent_probe[..., c:c + 1, m:m + 1, :, :],
                        axis=(-2, -1),
                    )

        # Use least-squares to find the optimal step sizes simultaneously
        # for all search directions.
        λ = 0.5
        if len(updates) == 1:  # (23)
            dX = updates[0]
            A = cp.sum(dX * dX.conj(), axis=(-2, -1))
            A += cp.mean(A) * λ
            b = cp.sum(cp.real(chi_ * dX.conj()), axis=(-2, -1))
            steps = (b / A)[..., None]
        elif len(updates) == 2:  # (22)
            A = cp.empty((*dOP.shape[:-2], 2, 2), dtype='complex64')
            A[..., 0, 0] = cp.sum(dOP * dOP.conj(), axis=(-2, -1))
            A[..., 0, 0] += cp.mean(A[..., 0, 0]) * λ
            A[..., 0, 1] = cp.sum(dOP * dPO.conj(), axis=(-2, -1))
            A[..., 1, 0] = A[..., 0, 1].conj()
            A[..., 1, 1] = cp.sum(dPO * dPO.conj(), axis=(-2, -1))
            A[..., 1, 1] += cp.mean(A[..., 0, 0]) * λ

            b = cp.empty((*dOP.shape[:-2], 2), dtype='complex64')
            b[..., 0] = cp.sum(cp.real(chi_ * dOP.conj()), axis=(-2, -1))
            b[..., 1] = cp.sum(cp.real(chi_ * dPO.conj()), axis=(-2, -1))
            steps = lstsq(A, b)

        num_steps = 0
        d = 0

        # Update each direction
        if recover_psi:
            step = steps[..., num_steps, None, None]
            num_steps += 1

            weighted_step = op.diffraction._patch(
                patches=step * probe_intensity,
                psi=cp.zeros_like(psi),
                scan=scan_,
                fwd=False,
            )

            psi += dir_psi * weighted_step / norm_probe
            d += step * dOP

        if recover_probe:
            step = steps[..., num_steps, None, None]
            num_steps += 1

            # Common probe update (27a)
            weighted_step = cp.sum(step * psi_intensity, axis=1, keepdims=True)
            probe_ += dir_probe * weighted_step / norm_psi

            d += step * dPO

        if __debug__:
            logger.info('%10s cost is %+12.5e', 'nearplane',
                        cp.linalg.norm(cp.ravel(chi_ - d)))

    if recover_probe and probe.shape[-3] > 1:
        probe = orthogonalize_eig(probe)

    result = {
        'psi': [psi],
        'probe': [probe],
        'cost': cost,
        'scan': scan,
    }
    if coherent_probe is not None:
        result['coherent_probe'] = [coherent_probe]
        result['weights'] = [weights]
    return result
