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
        super().__init__()
        self.propagation = propagation()
        self.diffraction = diffraction()

    def forward(
        self,
        psi: torch.Tensor,
        scan: torch.Tensor,
        probe: torch.Tensor,
        eigen: torch.Tensor = None,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        probes :  (   1, EIGEN, SHARED, WIDE, HIGH)
            The complex illumination function.
        weights : (POSI,     1, SHARED,    1,    1)
            Varies the contribution of each shared probe with scan position
        eigen :   (POSI, EIGEN,      1,    1,    1)
            Varies the contribution of each eigen probe with scan position
        psi : (WIDE, HIGH) complex64
            The wavefront modulation coefficients of the object.
        scan : (POSI, 2) float32
            Coordinates of the minimum corner of the probe grid for each
            measurement in the coordinate system of psi. Coordinate order
            consistent with WIDE, HIGH order.

        Returns
        -------
        intensity: (POSI, WIDE, HIGH) complex64
            The measured intensity hitting the detector respectively.

        """
        weighted_probes = torch.sum(
            probe, # * torch.tanh(eigen),
            dim=1,
        )
        farplane = self.propagation(self.diffraction(
                psi,
                scan,
                weighted_probes,
            ))
        return torch.sum(
            (farplane * torch.conj(farplane)).real,
            dim=1,
        )

