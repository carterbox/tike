__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2023, UChicago Argonne, LLC."

import torch


class Shift(torch.nn.Module):
    """Shift last two dimensions of an array using the Fourier method."""

    def forward(
        self,
        array: torch.Tensor,
        shift: torch.Tensor,
        scale: int = 2,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        array (..., H, W)
            The array to be shifted.
        shift (..., 2)
            The the shifts to be applied along the last two axes.

        Returns
        -------
        shifted (..., H, W)
            The shifted array.

        """
        padded = torch.fft.fft2(
            array,
            dim=(-2, -1),
            norm='ortho',
            s=(array.shape[-2]*scale, array.shape[-1]*scale),
        )
        freq0 = torch.fft.fftfreq(padded.shape[-2], device=padded.device)[..., None]
        padded = padded * torch.exp(-2j * torch.pi * freq0 * shift[..., 0, None, None])
        freq1 = torch.fft.fftfreq(padded.shape[-1], device=padded.device)[None, ...]
        padded = padded * torch.exp(-2j * torch.pi * freq1 * shift[..., 1, None, None])
        return torch.fft.ifft2(
            padded,
            dim=(-2, -1),
            norm='ortho',
        )[..., :array.shape[-2], :array.shape[-1]]
