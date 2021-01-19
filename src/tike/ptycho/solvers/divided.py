import logging

import cupy as cp

from tike.linalg import lstsq, projection, norm

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
    # TODO: Refactor so no using non-public API (_patch)
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

    # Solve the farplane phase problem ----------------------------------------
    farplane = op.propagation.fwd(nearplane, overwrite=True)
    intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
    cost = op.propagation.cost(data_, intensity)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)
    farplane -= 0.5 * op.propagation.grad(data_, farplane, intensity)

    if __debug__:
        intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
        cost = op.propagation.cost(data_, intensity)
        logger.info('%10s cost is %+12.5e', 'farplane', cost)
        # TODO: Only compute cost every 20 iterations or on a log sampling?

    farplane = op.propagation.adj(farplane, overwrite=True)

    # Solve the nearplane problem ---------------------------------------------
    # To find optimal step using least-squares we will convert from complex to
    # float and flatten the last two dimensions of the nearplanes
    lstsq_shape = (*nearplane.shape[:-3], 1, (pad - end) * (pad - end) * 2)

    for m in range(probe.shape[-3]):

        nearplane = farplane[..., m:m + 1, pad:end, pad:end]

        cprobe = common_probe[..., m:m + 1, :, :]
        uprobe = unique_probe[..., m:m + 1, :, :]

        patches = op.diffraction._patch(
            patches=cp.zeros(nearplane.shape, dtype='complex64'),
            psi=psi,
            scan=scan_,
            fwd=True,
        )

        # χ (diff) is the target for the nearplane problem; the difference
        # between the desired nearplane and the current nearplane that we wish
        # to minimize.
        diff = nearplane - uprobe * patches

        logger.info('%10s cost is %+12.5e', 'nearplane', norm(diff))

        updates = []

        if recover_psi:
            grad_psi = cp.conj(uprobe) * diff

            # Weight object updates by total probe intensity in each view
            total_illumination = uprobe * uprobe.conj()
            total_illumination /= norm(
                total_illumination,
                axis=(-5, -4, -3, -2, -1),
                keepdims=True,
            )
            assert total_illumination.dtype == 'complex64'

            # (25b) Common object gradient. Use a weighted (normalized) sum
            # instead of division as described in publication to improve
            # numerical stability.
            common_grad_psi = op.diffraction._patch(
                patches=grad_psi * total_illumination,
                psi=cp.zeros(psi.shape, dtype='complex64'),
                scan=scan_,
                fwd=False,
            )

            dOP = op.diffraction._patch(
                patches=cp.zeros(patches.shape, dtype='complex64'),
                psi=common_grad_psi,
                scan=scan_,
                fwd=True,
            ) * uprobe

            updates.append(dOP.view('float32').reshape(lstsq_shape))

        if recover_probe:
            grad_probe = xp.conj(patches) * diff

            # Weight probes in areas of higher transmission as more important
            total_transmission = patches * patches.conj()
            total_transmission /= norm(
                total_transmission,
                axis=(-5, -4, -3, -2, -1),
                keepdims=True,
            )

            # (25a) Common probe gradient. Use a weighted (normalized) sum
            # instead of division as described in publication to improve
            # numerical stability.
            common_grad_probe = cp.sum(
                grad_probe * total_transmission,
                axis=-5,
                keepdims=True,
            )

            dPO = common_grad_probe * patches
            updates.append(dPO.view('float32').reshape(lstsq_shape))

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
                phi = patches * coherent_probe[..., c:c + 1, m:m + 1, :, :]
                n = cp.mean(
                    cp.real(diff * phi.conj()),
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
        steps = lstsq(
            a=cp.stack(updates, axis=-1),
            b=diff.view('float32').reshape(lstsq_shape),
        )

        num_steps = 0

        # Update each direction
        if recover_psi:
            step = steps[..., num_steps, None, None]
            num_steps += 1

            # (27b) Object update
            weighted_step = op.diffraction._patch(
                patches=step * total_illumination,
                psi=cp.zeros(psi.shape, dtype='complex64'),
                scan=scan_,
                fwd=False,
            )

            psi += weighted_step * common_grad_psi

        if recover_probe:
            step = steps[..., num_steps, None, None]
            num_steps += 1

            # (27a) Probe update
            cprobe += common_grad_probe * cp.sum(
                step * total_transmission,
                axis=-5,
                keepdims=True,
            )

        if __debug__:
            patches = op.diffraction._patch(
                patches=cp.zeros(nearplane.shape, dtype='complex64'),
                psi=psi,
                scan=scan_,
                fwd=True,
            )
            logger.info('%10s cost is %+12.5e', 'nearplane',
                        norm(cprobe * patches - nearplane))

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
