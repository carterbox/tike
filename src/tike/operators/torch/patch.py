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
        positions : (N, 2)
            Coordinates of the minimum corner of the patches in the image grid.
        width :
            The width of the patches.

        Returns
        -------
        patches : (N, width, width)
            The extracted patches
        """
        patches = torch.empty_like(
            images,
            shape=(len(positions), *widths),
        )
        for i in range(len(positions)):
            patches[i] = images[
                positions[i, 0]:positions[i, 0] + widths[0],
                positions[i, 1]:positions[i, 1] + widths[1],
            ]
