"""Provide ptychography solvers and tooling."""
from enum import IntEnum

from .ptycho import *
from .position import check_allowed_positions


class Axis(IntEnum):
    """Provide names for the ordred dimensions in ptycho module arrays.

    Examples
    --------

    Get the probes from the 2nd incoherent mode:
    ```
    np.take(probes, (2, ), axis=Axis.MODE)
    ```

    Sum the frames for each view:
    ```
    np.sum(data, axis=Axis.FRAME, keepdims=True)
    ```

    Attributes
    ----------
    VIEW : int
        number of contiguous fields of view
    FRAME : int
        number of unique exposures on the detector
    POSITION : int
        unique positions during one exposure
    MODE : int
        number of incoherent probe modes
    WIDE : int
        number of pixels wide
    HIGH : int
        number of pixels high
    """
    VIEW = 0
    FRAME = 1
    POSITION = 2
    MODE = 3
    WIDE = 4
    HIGH = 5
