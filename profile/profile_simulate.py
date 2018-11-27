#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Benchmark ptychotomography reconstruction.

Profile tike.simulate and tike.admm on the function level by running the main
function of this script. Line by line profile hotspots for the file
tike/foo.py can be obtained by using pprofile. As below:

```
$ pprofile --statistic 0.001 --include tike/foo.py profile_admm.py
```
"""

import logging
import lzma
import os
import pickle
import pstats
from pyinstrument import Profiler
import unittest
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import tike
import numpy as np
import scipy
import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setUp(
        obj_file='../tests/data/nalm256.pickle.lzma',
        filename='./data/acquisition-params.hdf5',
        nview=720//4,
        pw=128//4,
        energy=5,
        voxelsize=1e-7,
        frames_per_view=256,
):
    """Set acquisition parameters."""
    if not os.path.isfile(filename):
        # Load a 3D object.
        with lzma.open(obj_file, 'rb') as file:
            obj = pickle.load(file).astype(np.complex64)
            # r = scipy.ndimage.zoom(obj.real, 2, order=0, mode='constant',
            #                        prefilter=False)
            # i = scipy.ndimage.zoom(obj.imag, 2, order=0, mode='constant',
            #                        prefilter=False)
            # obj = np.empty(r.shape, dtype=np.complex64)
            # obj.real = r
            # obj.imag = i
        # Create a probe.
        weights = tike.ptycho.gaussian(pw, rin=0.8, rout=1.0)
        probe = weights * np.exp(1j * weights * 0.2)
        # Define trajectory
        theta = np.linspace(0, np.pi,
                            num=nview,
                            endpoint=False)
        n = np.sqrt(frames_per_view).astype(np.int32)
        none, v, h = np.meshgrid(np.arange(theta.size),
                                 np.linspace(0, obj.shape[0]-pw, n),
                                 np.linspace(0, obj.shape[2]-pw-1, n),
                                 indexing='ij')
        for i in range(theta.size):
            h[i] += i / theta.size
        detector_shape = np.ones(2, dtype=np.int32) * pw * 2
        logging.info("""
        obj shape is {}
        probe size is {}
        nviews is {}
        frames per view is {}
        detector shape is {}
        """.format(obj.shape, probe.shape, nview, v[0].size, detector_shape))
        with h5py.File(filename, 'x') as file:
            file.create_dataset('implements',
                                data='exchange:measurement:process')
            set_experimenter_info(file)
            set_acquisition(
                file,
                obj, voxelsize,
                probe, energy,
                theta, v, h,
                detector_shape,
            )


def set_experimenter_info(f, info=None):
    """Add experimenter info the hdf5 file."""
    if info is None:
        info = {
            'name': 'Daniel Ching',
            'affiliation': 'Argonne National Laboratory',
            'address': '9700 S Cass Ave, Lemont, IL, 60439, USA',
            'email': 'dching@anl.gov',
        }
    for attribute in info:
        f.create_dataset('measurement/sample/experimenter/' + attribute,
                         data=info[attribute])


def set_acquisition(
        f,
        obj, voxelsize,
        probe, energy,
        theta, v, h,
        detector_shape,
):
    """Set the acquisition parameters into a dxchange hdf5."""
    f.create_dataset('measurement/sample/refractive_indices', data=obj)
    f['measurement/sample/refractive_indices'].attrs['voxelsize'] = voxelsize
    f.create_dataset('measurement/instrument/probe/function', data=probe)
    f['measurement/sample/refractive_indices'].attrs['energy'] = energy
    f.create_dataset('process/acquisition/image_theta', data=theta)
    f.create_dataset('process/acquisition/sample_image_shift_v', data=v)
    f.create_dataset('process/acquisition/sample_image_shift_h', data=h)
    f.create_dataset('measurement/instrument/detector/dimension_v',
                     data=detector_shape[0])
    f.create_dataset('measurement/instrument/detector/dimension_h',
                     data=detector_shape[1])


def test_simulate(acqu_filename='./data/acquisition-params.hdf5',
                  data_filename='./data/data.hdf5'):
    """Use pyinstrument to benchmark tike.simulate on one core."""
    comm = tike.MPICommunicator()
    with h5py.File(acqu_filename, 'r', driver='mpio', comm=comm.comm) as f:
        (
            obj, voxelsize,
            probe, energy,
            theta, v, h,
            detector_shape,
        ) = comm.get_acquisition(f=f)
    data = tike.simulate(
        obj=obj, voxelsize=voxelsize,
        probe=probe, theta=theta, v=v, h=h, energy=energy,
        detector_shape=detector_shape,
        comm=comm,
    )
    with h5py.File(data_filename, 'a', driver='mpio', comm=comm.comm) as f:
        comm.save_hdf5_distributed(f=f, path='exchange/data', data=data)
    logger.info("Node {} complete.".format(comm.rank))


if __name__ == '__main__':
    setUp()
    test_simulate()
    # setUp(
    #         obj_file='../tests/data/nalm256.pickle.lzma',
    #         filename='./data/large-acquisition-params.pickle',
    #         nview=720,
    #         pw=128,
    #         energy=5,
    #         voxelsize=1e-7,
    #         frames_per_view=1024,
    # )
    # test_simulate(acqu_filename='./data/large-acquisition-params.pickle',
    #               data_filename='./data/large-data.hdf5')
