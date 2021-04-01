# installation of dsm slam: https://github.com/jzubizarreta/dsm
# built on top of the setup_ORB file 

# run the setup_ORB first
# sudo ./setup_ORB.sh

###################
# cares solver
##################
git clone https://ceres-solver.googlesource.com/ceres-solver

# glog & gflags
sudo apt-get install libgoogle-glog-dev

# BLAS & LAPACK
sudo apt-get install libatlas-base-dev

# SuiteSparse
sudo apt-get install libsuitesparse-dev

# install
cd ceres-solver
mkdir build
cd build
cmake ..
make -j4
sudo make install	

cd ..