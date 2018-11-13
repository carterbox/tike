# Installation Instructions

At this time, `tike` can only be installed from source.

## Installation from the source code

`tike` installation has two steps: building the c extension shared library and installing the python package.

First, build the shared library by using the build script:

```
$ python build.py
```

This will compile a shared library, `libtike`(`.so` / `.dylib` / `.dll`) (Linux, MacOS, Windows) and install it to `tike/sharedlibs`.

For Windows, both MinGW-64 and `make` need to be installed and placed in the PATH, so that the appropriate `gcc.exe` (that is, one that supports C99) and `make.exe` can be found. For Anaconda Python on Windows, adding the `conda` packages `MinGW` and `make` provides these resources.

Next, install the python package to your current environment by using `pip`:

```
$ pip install -e .
```

We recommend `pip` for installation because it includes metadata with the installation that makes it easy to uninstall or update the package later. Calling `setup.py` directly does not create this metadata. The `-e` option for `pip install` makes the installation editable; this means whenever you import `tike`, any changes that you make to the source code will be included.

### Building parallel hdf5

There is no parallel hdf5 package available from conda-forge yet. Thus, you must build dxchange, hdf5, h5py, and mpi4py from source, so that they are built using the same mpi implementation.


#### Identify your MPI implementation.

If you don't already have an MPI implementation. Use one from from `conda`. If this is the route you choose, you can also install pre-built mpi4py.

```
$ conda install [openmpi, mpich] mpi4py
```

If you already have an MPI implementation find the MPI compiler, `mpicc`.

#### HDF5

Download the (hdf5 source code)[https://www.hdfgroup.org/downloads/hdf5/source-code/] and follow the directions for installation using `./configure`. The following flags must be included, and you must use the MPI compiler.

```
$ export CC=mpicc
$ ./configure --enable-parallel --enable-shared --prefix=$CONDA_PREFIX
$ make install
```

#### h5py

Download the h5py source and follow the directions to (build from source)[http://docs.h5py.org/en/stable/mpi.html].

```
$ export CC=mpicc
$ python setup.py configure --mpi --hdf5=$CONDA_PREFIX
$ python setup.py build
```

#### dxchange and mpi4py

dxchange requires h5py, so it must be installed from source after h5py. Installation of either of these two using conda will cause the conda the installation of dependencies which will shadow the ones you just built.
