"""Defines a free-space propagation operator based on the CuPy FFT module."""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2023, UChicago Argonne, LLC."

import torch

class Propagation(torch.nn.Module):
    """A Fourier-based free-space propagation using CuPy."""

    def forward(
        self,
        nearplane: torch.Tensor,
    ) -> torch.Tensor:
        """Forward Fourier-based free-space propagation operator.

        Parameters
        ----------
        nearplane: (..., detector_shape, detector_shape) complex64
            The wavefronts after exiting the object.

        """
        return torch.fft.fft2(
            nearplane,
            norm='ortho',
            dims=(-2, -1),
        )
