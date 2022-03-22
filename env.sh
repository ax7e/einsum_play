# wget https://developer.download.nvidia.com/compute/cutensor/1.3.1/local_installers/libcutensor-linux-x86_64-1.3.1.3.tar.gz
# tar xvf libcutensor-linux-x86_64-1.3.1.3.tar.gz 
export CUTENSOR_ROOT=`pwd`/libcutensor
# sudo apt install nvidia-cuda-toolkit
export LD_LIBRARY_PATH=${CUTENSOR_ROOT}/lib/10.1/:${LD_LIBRARY_PATH}

nvcc trival.cu -L${CUTENSOR_ROOT}/lib/11.0/ -I${CUTENSOR_ROOT}/include -std=c++11 -lcutensor -o contraction

