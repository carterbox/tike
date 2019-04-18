"""Define an communication class to move data between processes."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['MPICommunicator']

import logging
import pickle

import h5py
from mpi4py import MPI
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MPICommunicator(object):
    """Communicate between processes using MPI.

    Use this class to astract away all of the MPI communication that needs to
    occur in order to switch between the tomography and ptychography problems.
    """

    def __init__(self):
        """Load the MPI params and get initial data."""
        super(MPICommunicator, self).__init__()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        logger.info("Node {:,d} is running.".format(self.rank))

    def scatter(self, *args):
        """Send and recieve constant data that must be divided."""
        if len(args) == 1:
            arg = args[0]
            if self.rank == 0:
                chunks = np.array_split(arg, self.size)
            else:
                chunks = None
            return self.comm.scatter(chunks, root=0)
        out = list()
        for arg in args:
            if self.rank == 0:
                chunks = np.array_split(arg, self.size)
            else:
                chunks = None
            out.append(self.comm.scatter(chunks, root=0))
        return out

    def broadcast(self, *args):
        """Synchronize parameters that are the same for all processses."""
        if len(args) == 1:
            return self.comm.bcast(args[0], root=0)
        out = list()
        for arg in args:
            out.append(self.comm.bcast(arg, root=0))
        return out

    def get_ptycho_slice(self, tomo_slice):
        """Switch to slicing for the pytchography problem."""
        # Break the tomo data along the theta axis
        t_chunks = np.array_split(tomo_slice, self.size, axis=0)  # Theta, V, H
        # Each rank takes a turn scattering its tomo v slice to the others
        p_chunks = list()
        for i in range(self.size):
            p_chunks.append(self.comm.scatter(t_chunks, root=i))
        # Recombine the along vertical axis so each rank now has a theta slice
        return np.concatenate(p_chunks, axis=1)  # Theta, V, H

    def get_tomo_slice(self, ptych_slice):
        """Switch to slicing for the tomography problem."""
        # Break the ptych data along the vertical axis
        p_chunks = np.array_split(ptych_slice, self.size, axis=1)
        # Each rank takes a turn scattering its ptych theta slice to the others
        t_chunks = list()
        for i in range(self.size):
            t_chunks.append(self.comm.scatter(p_chunks, root=i))
        # Recombine along the theta axis so each rank now has a vertical slice
        return np.concatenate(t_chunks, axis=0)  # Theta, V, H

    def gather(self, arg, root=0, axis=0):
        """Gather arg to one node."""
        arg = self.comm.gather(arg, root=root)
        if self.rank == root:
            return np.concatenate(arg, axis=axis)
        return None

    def allgather(self, arg, axis=0):
        """All nodes gather arg."""
        return self.comm.allgather(arg)

    def get_acquisition(self, f):
        """Get the acquisition parameters from a dxchange hdf5 file.

        Parameters
        ----------
        f : hdf5 file context
        comm : MPICommunicator

        """
        obj_path = 'measurement/sample/refractive_indices'
        obj = self.load_hdf5_distributed(f=f, path=obj_path)
        voxelsize = f[obj_path].attrs['voxelsize']
        energy = f[obj_path].attrs['energy']
        probe = f['measurement/instrument/probe/function'].value
        theta = f['process/acquisition/image_theta'].value
        v = self.load_hdf5_distributed(
            f=f, path='process/acquisition/sample_image_shift_v')
        h = self.load_hdf5_distributed(
            f=f, path='process/acquisition/sample_image_shift_h')
        detector_shape = (
            f['measurement/instrument/detector/dimension_v'].value,
            f['measurement/instrument/detector/dimension_h'].value,
        )
        return (
            obj, voxelsize,
            probe, energy,
            theta, v, h,
            detector_shape,
        )  # yapf: disable

    def load_hdf5_distributed(self, f, path):
        """Load data from an hdf5 file distributed along axis zero.

        Parameters
        ----------
        f : hdf5 file context
        path : string
            The path to the data in the hdf file.

        """
        div_points = chunk_indices(f[path].shape[0], self.size)
        return f[path][div_points[self.rank]:div_points[self.rank + 1], ...]

    def save_hdf5_distributed(self, f, path, data):
        """Save data to an hdf5 file from data distributed along axis zero.

        Parameters
        ----------
        f : hdf5 file context
        path : string
            The path to the data in the hdf file.
        data : array-like

        """
        # Determine the full shape and location of the data
        data = np.asarray(data)
        chunk_sizes = self.comm.allgather(len(data))
        lo = np.sum(chunk_sizes[:self.rank], dtype=int)
        hi = lo + len(data)
        combined_shape = (np.sum(chunk_sizes), *data[0].shape)
        logger.info("\nThe combined shape is {}."
                    "\nThis chunk range {}.".format(combined_shape, (lo, hi)))
        # Compute data and write to file
        f.create_dataset(path, shape=combined_shape, dtype=data.dtype)
        f[path][lo:hi, ...] = data


def chunk_indices(Ntotal, Nsections):
    """Return division indices from breaking Ntotal things into Nsections."""
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = ([0] + extras * [Neach_section + 1] +
                     (Nsections - extras) * [Neach_section])
    div_points = np.cumsum(section_sizes)
    return div_points
