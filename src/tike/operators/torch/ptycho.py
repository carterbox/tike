"""Defines a ptychography operator based on pytorch Modules"""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2023, UChicago Argonne, LLC."

import typing

import torch

from .propagation import *
from .diffraction import *


class Ptycho(torch.nn.Module):
    """A Ptychography operator.

    Compose a diffraction and propagation operator to simulate the interaction
    of an illumination wavefront with an object followed by the propagation of
    the wavefront to a detector plane.

    .. versionadded:: 0.26.0

    """

    def __init__(
        self,
        propagation: torch.nn.Module = Propagation,
        diffraction: torch.nn.Module = OpticallyThinDiffraction,
    ):
        super.__init__()
        self.propagation = propagation()
        self.diffraction = diffraction()

    def forward(
        self,
        psi: torch.Tensor,
        scan: torch.Tensor,
        probe: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        probes :  (   1,    1, SHARED, WIDE, HIGH)
            The complex illumination function.
        weights : (POSI,    1, SHARED,    1,    1)
            Varies the contribution of each shared probe with scan position
        psi : (WIDE, HIGH) complex64
            The wavefront modulation coefficients of the object.
        scan : (POSI, 2) float32
            Coordinates of the minimum corner of the probe grid for each
            measurement in the coordinate system of psi. Coordinate order
            consistent with WIDE, HIGH order.


        Returns
        -------
        farplane: (..., POSI, 1, SHARED, detector_shape, detector_shape) complex64
            The wavefronts hitting the detector respectively.

        """
        if weights:
            # restrict weights to [-1, 1] range
            weighted_probes = probe * torch.tanh(weights)
        else:
            weighted_probes = probe
        return self.propagation(self.diffraction(
            psi,
            scan,
            weighted_probes,
        ))
