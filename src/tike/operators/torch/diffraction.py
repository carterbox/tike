__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2023, UChicago Argonne, LLC."

import torch

from .patch import *
from .shift import *


class OpticallyThinDiffraction(torch.nn.Module):
    """Simulates optically thin diffraction (multiplication)."""

    def __init__(self):
        super().__init__()
        self.patch = Patch()
        self.shift = Shift()

    def forward(
        self,
        psi: torch.Tensor,
        scan: torch.Tensor,
        kernels: torch.Tensor,
    ):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.

        Parameters
        ----------
        psi : (H, W)
            The complex wavefront modulation of the object.
        scan : (nscan, 2)
            Coordinates of the minimum corner of the kernels grid for each
            measurement in the coordinate system of psi. Vertical coordinates
            first, horizontal coordinates second.
        kernels :
            The (nscan, nkernels, H1, W1) or (1, nkernels, H1, W1) complex
            illumination function.

        """
        assert psi.ndim == 2, psi.shape
        assert scan.ndim == 2, scan.shape
        assert scan.shape[-1] == 2, scan.shape
        assert kernels.ndim == 4, kernels.shape
        assert kernels.shape[-4] == 1 or kernels.shape[-4] == scan.shape[-2]

        integer = scan.int()
        noninteger = integer - scan
        patches = self.patch(
            images=psi,
            positions=integer,
            widths=kernels.shape[-2:],
        )
        patches1 = self.shift(
            array=patches,
            shift=noninteger,
        )
        # kernels1 = self.shift(
        #     array=kernels,
        #     shift=-noninteger[:, None, ...],
        # )
        return kernels * patches1[:, None, ...]
