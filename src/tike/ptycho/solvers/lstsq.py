import logging
import typing

import cupy as cp
import numpy.typing as npt

import tike.communicators
import tike.linalg
import tike.operators
import tike.opt
import tike.ptycho.position
import tike.ptycho.probe
import tike.ptycho.object
import tike.precision

from .options import *

logger = logging.getLogger(__name__)


def lstsq_grad(
    op: tike.operators.Ptycho,
    comm: tike.communicators.Comm,
    data: typing.List[npt.NDArray],
    batches: typing.List[npt.NDArray[cp.intc]],
    *,
    parameters: PtychoParameters,
):
    """Solve the ptychography problem using Odstrcil et al's approach.

    Object and probe are updated simultaneously using optimal step sizes
    computed using a least squares approach.

    Parameters
    ----------
    op : :py:class:`tike.operators.Ptycho`
        A ptychography operator.
    comm : :py:class:`tike.communicators.Comm`
        An object which manages communications between GPUs and nodes.
    data : list((FRAME, WIDE, HIGH) float32, ...)
        A list of unique CuPy arrays for each device containing
        the intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records. FFT-shifted so the
        diffraction peak is at the corners.
    batches : list(list((BATCH_SIZE, ) int, ...), ...)
        A list of list of indices along the FRAME axis of `data` for
        each device which define the batches of `data` to process
        simultaneously.
    parameters : :py:class:`tike.ptycho.solvers.PtychoParameters`
        An object which contains reconstruction parameters.

    Returns
    -------
    result : dict
        A dictionary containing the updated keyword-only arguments passed to
        this function.

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iterative
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.

    .. seealso:: :py:mod:`tike.ptycho`

    """
    probe = parameters.probe
    scan = parameters.scan
    psi = parameters.psi
    algorithm_options = parameters.algorithm_options
    probe_options = parameters.probe_options
    position_options = parameters.position_options
    object_options = parameters.object_options
    eigen_probe = parameters.eigen_probe
    eigen_weights = parameters.eigen_weights

    position_update_numerator = [None] * comm.pool.num_workers
    position_update_denominator = [None] * comm.pool.num_workers

    if eigen_probe is None:
        beigen_probe = [None] * comm.pool.num_workers
    else:
        beigen_probe = eigen_probe

    if object_options is not None:
        if algorithm_options.batch_method == 'compact':
            object_options.combined_update = cp.zeros_like(psi[0])

    if probe_options is not None:
        probe_options.probe_update_sum = cp.zeros_like(probe[0])

    if parameters.algorithm_options.batch_method == 'compact':
        order = range
    else:
        order = tike.opt.randomizer.permutation

    batch_cost = []
    beta_object = []
    beta_probe = []
    for n in order(len(batches[0])):

        bdata = comm.pool.map(tike.opt.get_batch, data, batches, n=n)

        if eigen_weights is None:
            unique_probe = probe
            beigen_weights = [None] * comm.pool.num_workers
        else:
            beigen_weights = comm.pool.map(
                tike.opt.get_batch,
                eigen_weights,
                batches,
                n=n,
            )
            unique_probe = comm.pool.map(
                tike.ptycho.probe.get_varying_probe,
                probe,
                beigen_probe,
                beigen_weights,
            )

        (
            psi,
            probe,
            beigen_probe,
            beigen_weights,
            scan,
            bbeta_object,
            bbeta_probe,
            costs,
            position_update_numerator,
            position_update_denominator,
        ) = _update_nearplane(
            op,
            comm,
            bdata,
            psi,
            scan,
            probe,
            unique_probe,
            beigen_probe,
            beigen_weights,
            batches,
            position_update_numerator,
            position_update_denominator,
            recover_psi=object_options is not None,
            recover_probe=probe_options is not None,
            recover_positions=position_options is not None,
            n=n,
            num_batch=algorithm_options.num_batch,
            psi_update_denominator=object_options.preconditioner,
            object_options=object_options,
            probe_options=probe_options,
            algorithm_options=algorithm_options,
        )

        for c in costs:
            batch_cost = batch_cost + c.tolist()

        beta_object.append(bbeta_object)
        beta_probe.append(bbeta_probe)

        if eigen_weights is not None:
            comm.pool.map(
                tike.opt.put_batch,
                beigen_weights,
                eigen_weights,
                batches,
                n=n,
            )

    if position_options:
        scan, position_options = zip(*comm.pool.map(
            _update_position,
            scan,
            position_options,
            position_update_numerator,
            position_update_denominator,
        ))

    algorithm_options.costs.append(batch_cost)

    if object_options and algorithm_options.batch_method == 'compact':
        object_update_precond = _precondition_object_update(
            object_options.combined_update,
            object_options.preconditioner[0],
        )

        # (27b) Object update
        beta_object = cp.mean(cp.stack(beta_object))
        dpsi = beta_object * object_update_precond
        psi[0] = psi[0] + dpsi

        if object_options.use_adaptive_moment:
            (
                dpsi,
                object_options.v,
                object_options.m,
            ) = _momentum_checked(
                g=dpsi,
                v=object_options.v,
                m=object_options.m,
                mdecay=object_options.mdecay,
                errors=list(np.mean(x) for x in algorithm_options.costs[-3:]),
                beta=beta_object,
                memory_length=3,
            )
            weight = object_options.preconditioner[0]
            weight = weight / (0.1 * weight.max() + weight)
            psi[0] = psi[0] + weight * dpsi

        psi = comm.pool.bcast([psi[0]])

    if probe_options:
        if probe_options.use_adaptive_moment:
            beta_probe = cp.mean(cp.stack(beta_probe))
            dprobe = probe_options.probe_update_sum
            if probe_options.v is None:
                probe_options.v = np.zeros_like(
                    dprobe,
                    shape=(3, *dprobe.shape),
                )
            if probe_options.m is None:
                probe_options.m = np.zeros_like(dprobe,)
            # ptychoshelves only applies momentum to the main probe
            mode = 0
            (
                d,
                probe_options.v[..., mode, :, :],
                probe_options.m[..., mode, :, :],
            ) = _momentum_checked(
                g=dprobe[..., mode, :, :],
                v=probe_options.v[..., mode, :, :],
                m=probe_options.m[..., mode, :, :],
                mdecay=probe_options.mdecay,
                errors=list(np.mean(x) for x in algorithm_options.costs[-3:]),
                beta=beta_probe,
                memory_length=3,
            )
            probe[0][..., mode, :, :] = probe[0][..., mode, :, :] + d
            probe = comm.pool.bcast([probe[0]])

    parameters.probe = probe
    parameters.psi = psi
    parameters.scan = scan
    parameters.algorithm_options = algorithm_options
    parameters.probe_options = probe_options
    parameters.object_options = object_options
    parameters.position_options = position_options
    parameters.eigen_weights = eigen_weights
    parameters.eigen_probe = eigen_probe
    return parameters


def _update_nearplane(
    op: tike.operators.Ptycho,
    comm: tike.communicators.Comm,
    data_: typing.List[npt.NDArray],
    psi: typing.List[npt.NDArray[cp.csingle]],
    scan: typing.List[npt.NDArray[cp.single]],
    probe: typing.List[npt.NDArray[cp.csingle]],
    unique_probe: typing.List[npt.NDArray[cp.csingle]],
    eigen_probe: typing.List[npt.NDArray[cp.csingle]],
    eigen_weights: typing.List[npt.NDArray[cp.single]],
    batches,
    position_update_numerator,
    position_update_denominator,
    *,
    recover_psi: bool,
    recover_probe: bool,
    recover_positions: bool,
    psi_update_denominator: npt.NDArray[cp.csingle],
    num_batch: int,
    n: int,
    object_options: typing.Union[ObjectOptions, None],
    probe_options: typing.Union[ProbeOptions, None],
    algorithm_options: LstsqOptions,
):

    if True:
        (
            diff,
            probe_update,
            object_upd_sum,
            m_probe_update,
            costs,
            patches,
            position_update_numerator,
            position_update_denominator,
        ) = (list(a) for a in zip(*comm.pool.map(
            _get_nearplane_gradients,
            data_,
            psi,
            scan,
            unique_probe,
            batches,
            position_update_numerator,
            position_update_denominator,
            n=n,
            op=op,
            recover_psi=recover_psi,
            recover_probe=recover_probe,
            recover_positions=recover_positions,
        )))

        if recover_psi:
            object_upd_sum = comm.Allreduce(object_upd_sum)

        if recover_probe:
            m_probe_update = comm.Allreduce_mean(
                m_probe_update,
                axis=-5,
            )

        (
            object_update_precond,
            m_probe_update,
            A1,
            A2,
            A4,
            b1,
            b2,
        ) = (list(a) for a in zip(*comm.pool.map(
            _precondition_nearplane_gradients,
            diff,
            scan,
            unique_probe,
            probe,
            object_upd_sum,
            m_probe_update,
            psi_update_denominator,
            patches,
            batches,
            n=n,
            op=op,
            m=0,
            recover_psi=recover_psi,
            recover_probe=recover_probe,
            probe_options=probe_options,
        )))

        if recover_psi:
            A1_delta = comm.pool.bcast([comm.Allreduce_mean(A1, axis=-3)])

        if recover_probe:
            A4_delta = comm.pool.bcast([comm.Allreduce_mean(A4, axis=-3)])

        m = 0

        if m == 0 and (recover_probe or recover_psi):
            (
                weighted_step_psi,
                weighted_step_probe,
            ) = (list(a) for a in zip(*comm.pool.map(
                _get_nearplane_steps,
                A1,
                A2,
                A4,
                b1,
                b2,
                A1_delta,
                A4_delta,
                recover_psi=recover_psi,
                recover_probe=recover_probe,
                m=m,
            )))
            weighted_step_psi[0] = comm.Allreduce_mean(
                weighted_step_psi,
                axis=-5,
            )[..., 0, 0, 0]
            weighted_step_probe[0] = comm.Allreduce_mean(
                weighted_step_probe,
                axis=-5,
            )

        if m == 0 and recover_probe and eigen_weights[0] is not None:
            logger.info('Updating eigen probes')

            eigen_weights = comm.pool.map(
                _get_coefs_intensity,
                eigen_weights,
                diff,
                probe,
                patches,
                m=m,
            )

            # (30) residual probe updates
            if eigen_weights[0].shape[-2] > 1:
                R = comm.pool.map(
                    _get_residuals,
                    probe_update,
                    m_probe_update,
                    m=m,
                )

            if eigen_probe[0] is not None and m < eigen_probe[0].shape[-3]:
                assert eigen_weights[0].shape[
                    -2] == eigen_probe[0].shape[-4] + 1
                for n in range(1, eigen_probe[0].shape[-4] + 1):

                    (
                        eigen_probe,
                        eigen_weights,
                    ) = tike.ptycho.probe.update_eigen_probe(
                        comm,
                        R,
                        eigen_probe,
                        eigen_weights,
                        patches,
                        diff,
                        β=min(0.1, 1.0 / num_batch),
                        c=n,
                        m=m,
                    )

                    if n + 1 < eigen_weights[0].shape[-2]:
                        # Subtract projection of R onto new probe from R
                        R = comm.pool.map(
                            _update_residuals,
                            R,
                            eigen_probe,
                            axis=(-2, -1),
                            c=n - 1,
                            m=m,
                        )

        # Update each direction
        if object_options is not None:
            if algorithm_options.batch_method != 'compact':
                # (27b) Object update
                dpsi = weighted_step_psi[0] * object_update_precond[0]

                if object_options.use_adaptive_moment:
                    (
                        dpsi,
                        object_options.v,
                        object_options.m,
                    ) = tike.opt.momentum(
                        g=dpsi,
                        v=object_options.v,
                        m=object_options.m,
                        vdecay=object_options.vdecay,
                        mdecay=object_options.mdecay,
                    )
                psi[0] = psi[0] + dpsi
                psi = comm.pool.bcast([psi[0]])
            else:
                object_options.combined_update += object_upd_sum[0]

        if probe_options is not None:
            dprobe = weighted_step_probe[0] * m_probe_update[0]
            probe_options.probe_update_sum += dprobe / num_batch
            # (27a) Probe update
            probe[0] += dprobe
            probe = comm.pool.bcast([probe[0]])

    return (
        psi,
        probe,
        eigen_probe,
        eigen_weights,
        scan,
        weighted_step_psi[0],
        weighted_step_probe[0],
        costs,
        position_update_numerator,
        position_update_denominator,
    )


def _get_nearplane_gradients(
    data: npt.NDArray,
    psi: npt.NDArray[cp.csingle],
    scan: npt.NDArray[cp.single],
    unique_probe: npt.NDArray[cp.csingle],
    batches,
    position_update_numerator,
    position_update_denominator,
    *,
    n: int,
    op: tike.operators.Ptycho,
    recover_psi: bool,
    recover_probe: bool,
    recover_positions: bool,
):
    indices = batches[n]

    farplane = op.fwd(probe=unique_probe, scan=scan[indices], psi=psi)
    intensity = cp.sum(
        cp.square(cp.abs(farplane)),
        axis=list(range(1, farplane.ndim - 2)),
    )
    costs = getattr(tike.operators,
                    f'{op.propagation.model}_each_pattern')(data, intensity)
    cost = cp.mean(costs)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)
    farplane = -op.propagation.grad(data, farplane, intensity)

    farplane = op.propagation.adj(farplane, overwrite=True)

    pad, end = op.diffraction.pad, op.diffraction.end
    chi = farplane[..., pad:end, pad:end]

    # Get update directions for each scan positions
    if recover_psi:
        # (24b)
        object_update_proj = cp.conj(unique_probe) * chi
        # (25b) Common object gradient.
        object_upd_sum = op.diffraction.patch.adj(
            patches=object_update_proj.reshape(
                len(scan[indices]) * chi.shape[-3], *chi.shape[-2:]),
            images=cp.zeros_like(psi),
            positions=scan[indices],
            nrepeat=chi.shape[-3],
        )
    else:
        object_upd_sum = None

    if recover_probe:
        patches = op.diffraction.patch.fwd(
            patches=cp.zeros_like(chi[..., 0, 0, :, :]),
            images=psi,
            positions=scan[indices],
        )[..., None, None, :, :]
        # (24a)
        probe_update = cp.conj(patches) * chi
        # (25a) Common probe gradient. Use simple average instead of
        # division as described in publication because that's what
        # ptychoshelves does
        m_probe_update = cp.mean(
            probe_update,
            axis=-5,
            keepdims=True,
        )
    else:
        probe_update = None
        m_probe_update = None
        patches = None

    if recover_positions:
        m = 0
        position_update_numerator = cp.empty_like(
            scan,
        ) if position_update_numerator is None else position_update_numerator
        position_update_denominator = cp.empty_like(
            scan,
        ) if position_update_denominator is None else position_update_denominator

        grad_x, grad_y = tike.ptycho.position.gaussian_gradient(patches)

        position_update_numerator[indices, ..., 0] = cp.sum(
            cp.real(
                cp.conj(grad_x * unique_probe[..., [m], :, :]) *
                chi[..., [m], :, :]),
            axis=(-4, -3, -2, -1),
        )
        position_update_denominator[indices, ..., 0] = cp.sum(
            cp.abs(grad_x * unique_probe[..., [m], :, :])**2,
            axis=(-4, -3, -2, -1),
        )
        position_update_numerator[indices, ..., 1] = cp.sum(
            cp.real(
                cp.conj(grad_y * unique_probe[..., [m], :, :]) *
                chi[..., [m], :, :]),
            axis=(-4, -3, -2, -1),
        )
        position_update_denominator[indices, ..., 1] = cp.sum(
            cp.abs(grad_y * unique_probe[..., [m], :, :])**2,
            axis=(-4, -3, -2, -1),
        )

    return (
        chi,
        probe_update,
        object_upd_sum,
        m_probe_update,
        costs,
        patches,
        position_update_numerator,
        position_update_denominator,
    )


def _precondition_object_update(
    object_upd_sum: npt.NDArray[cp.csingle],
    psi_update_denominator: npt.NDArray[cp.csingle],
    alpha: float = 0.05,
) -> npt.NDArray[cp.csingle]:
    return object_upd_sum / cp.sqrt(
        cp.square((1 - alpha) * psi_update_denominator) +
        cp.square(alpha * psi_update_denominator.max(
            axis=(-2, -1),
            keepdims=True,
        )))


def _precondition_nearplane_gradients(
    nearplane,
    scan,
    unique_probe,
    probe,
    object_upd_sum,
    m_probe_update,
    psi_update_denominator,
    patches,
    batches,
    *,
    n: int,
    op,
    m,
    recover_psi,
    recover_probe,
    alpha=0.05,
    probe_options,
):
    indices = batches[n]

    eps = op.xp.float32(1e-9) / (nearplane.shape[-2] * nearplane.shape[-1])

    if recover_psi:
        object_update_precond = _precondition_object_update(
            object_upd_sum,
            psi_update_denominator,
        )

        object_update_proj = op.diffraction.patch.fwd(
            patches=cp.zeros_like(nearplane[..., 0, 0, :, :]),
            images=object_update_precond,
            positions=scan[indices],
        )
        dOP = object_update_proj[..., None,
                                 None, :, :] * unique_probe[..., [m], :, :]

        A1 = cp.sum((dOP * dOP.conj()).real + eps, axis=(-2, -1))
    else:
        object_update_proj = None
        dOP = None
        A1 = None

    if recover_probe:

        # b0 = tike.ptycho.probe.finite_probe_support(
        #     unique_probe[..., [m], :, :],
        #     p=probe_options.probe_support,
        #     radius=probe_options.probe_support_radius,
        #     degree=probe_options.probe_support_degree,
        # )

        # b1 = probe_options.additional_probe_penalty * cp.linspace(
        #     0,
        #     1,
        #     probe[0].shape[-3],
        #     dtype=tike.precision.floating,
        # )[..., [m], None, None]

        # m_probe_update = (m_probe_update -
        #                   (b0 + b1) * probe[..., [m], :, :]) / (
        #                       (1 - alpha) * probe_update_denominator +
        #                       alpha * probe_update_denominator.max(
        #                           axis=(-2, -1),
        #                           keepdims=True,
        #                       ) + b0 + b1)

        dPO = m_probe_update[..., [m], :, :] * patches
        A4 = cp.sum((dPO * dPO.conj()).real + eps, axis=(-2, -1))
    else:
        dPO = None
        A4 = None

    if recover_psi and recover_probe:
        b1 = cp.sum((dOP.conj() * nearplane[..., [m], :, :]).real,
                    axis=(-2, -1))
        b2 = cp.sum((dPO.conj() * nearplane[..., [m], :, :]).real,
                    axis=(-2, -1))
        A2 = cp.sum((dOP * dPO.conj()), axis=(-2, -1))
    elif recover_psi:
        b1 = cp.sum((dOP.conj() * nearplane[..., [m], :, :]).real,
                    axis=(-2, -1))
    elif recover_probe:
        b2 = cp.sum((dPO.conj() * nearplane[..., [m], :, :]).real,
                    axis=(-2, -1))

    return (
        object_update_precond,
        m_probe_update,
        A1,
        A2,
        A4,
        b1,
        b2,
    )


def _get_nearplane_steps(A1, A2, A4, b1, b2, A1_delta, A4_delta, recover_psi,
                         recover_probe, m):

    A1 += 0.5 * A1_delta
    A4 += 0.5 * A4_delta

    # (22) Use least-squares to find the optimal step sizes simultaneously
    if recover_psi and recover_probe:
        A3 = A2.conj()
        determinant = A1 * A4 - A2 * A3
        x1 = -cp.conj(A2 * b2 - A4 * b1) / determinant
        x2 = cp.conj(A1 * b2 - A3 * b1) / determinant
    elif recover_psi:
        x1 = b1 / A1
    elif recover_probe:
        x2 = b2 / A4
    else:
        x1 = None
        x2 = None

    if recover_psi:
        step = 0.9 * cp.maximum(0, x1[..., None, None].real)

        # (27b) Object update
        beta_object = cp.mean(step, keepdims=True, axis=-5)
    else:
        beta_object = None

    if recover_probe:
        step = 0.9 * cp.maximum(0, x2[..., None, None].real)

        beta_probe = cp.mean(step, axis=-5, keepdims=True)
    else:
        beta_probe = None

    return beta_object, beta_probe


def _get_coefs_intensity(weights, xi, P, O, m):
    OP = O * P[..., [m], :, :]
    num = cp.sum(cp.real(cp.conj(OP) * xi[..., [m], :, :]), axis=(-1, -2))
    den = cp.sum(cp.abs(OP)**2, axis=(-1, -2))
    weights[..., 0:1, [m]] = weights[..., 0:1, [m]] + 0.1 * num / den
    return weights


def _get_residuals(grad_probe, grad_probe_mean, m):
    return grad_probe[..., [m], :, :] - grad_probe_mean[..., [m], :, :]


def _update_residuals(R, eigen_probe, axis, c, m):
    R -= tike.linalg.projection(
        R,
        eigen_probe[..., c:c + 1, m:m + 1, :, :],
        axis=axis,
    )
    return R


def _update_position(
    scan: npt.NDArray,
    position_options: PositionOptions,
    position_update_numerator: npt.NDArray,
    position_update_denominator: npt.NDArray,
    *,
    alpha=0.05,
    max_shift=1,
):
    step = (position_update_numerator) / (
        (1 - alpha) * position_update_denominator +
        alpha * max(position_update_denominator.max(), 1e-6))

    if position_options.use_adaptive_moment:
        logger.info(
            "position correction with ADAptive Momemtum acceleration enabled.")
        (
            step,
            position_options.v,
            position_options.m,
        ) = tike.opt.adam(
            step,
            position_options.v,
            position_options.m,
            vdecay=position_options.vdecay,
            mdecay=position_options.mdecay,
        )

    scan -= step

    return scan, position_options


def _momentum_checked(
    g: npt.NDArray,
    v: typing.Union[None, npt.NDArray],
    m: typing.Union[None, npt.NDArray],
    mdecay: float,
    errors: typing.List[float],
    beta: float = 1.0,
    memory_length: int = 3,
    vdecay=None,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Momentum updates, but only if the cost function is trending downward.

    Parameters
    ----------
    previous_g (EPOCH, WIDTH, HEIGHT)
        The previous psi updates
    g (WIDTH, HEIGHT)
        The current psi update
    """
    m = np.zeros_like(g,) if m is None else m
    previous_g = np.zeros_like(
        g,
        shape=(memory_length, *g.shape),
    ) if v is None else v

    # Keep a running list of the update directions
    previous_g = np.roll(previous_g, shift=-1, axis=0)
    previous_g[-1] = g / tike.linalg.norm(g) * beta

    # Only apply momentum updates if the objective function is decreasing
    if (len(errors) > 2
            and max(errors[-3], errors[-2]) > min(errors[-2], errors[-1])):
        # Check that previous updates are moving in a similar direction
        previous_update_correlation = tike.linalg.inner(
            previous_g[:-1],
            previous_g[-1],
            axis=(-2, -1),
        ).real.flatten()
        if np.all(previous_update_correlation > 0):
            friction, _ = tike.opt.fit_line_least_squares(
                x=np.arange(len(previous_update_correlation) + 1),
                y=[
                    0,
                ] + np.log(previous_update_correlation).tolist(),
            )
            friction = 0.5 * max(-friction, 0)
            m = (1 - friction) * m + g
            return mdecay * m, previous_g, m

    return np.zeros_like(g), previous_g, m / 2
