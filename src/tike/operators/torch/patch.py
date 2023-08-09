__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2023, UChicago Argonne, LLC."

import typing
import torch


class Patch(torch.nn.Module):
    """Extract patches from images at provided positions."""

    def forward(
        self,
        images: torch.Tensor,
        positions: torch.Tensor,
        widths: typing.Tuple[int, int],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        images : (H, W)
            The complex wavefront modulation of the object.
        positions : (N, 2) int
            Coordinates of the minimum corner of the patches in the image grid.
        width :
            The width of the patches.

        Returns
        -------
        patches : (N, width, width)
            The extracted patches
        """
        if positions.is_floating_point():
            msg = (
                "The `positions` argument of the Patch operator must be of "
                f"integer type; not {positions.dtype}"
            )
            raise ValueError(msg)
        patches = torch.empty(
            (len(positions), *widths),
            dtype=images.dtype,
            layout=images.layout,
            device=images.device,
        )
        for i in range(len(positions)):
            patches[i] = images[
                positions[i, 0]:positions[i, 0] + widths[0],
                positions[i, 1]:positions[i, 1] + widths[1],
            ]
        return patches
