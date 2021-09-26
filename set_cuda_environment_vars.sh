#!/bin/bash
# Simple script to install CUDA Toolkit and cuDNN
#
# @author: Bruce Goldfeder
# @advisor: Dr. Igor Griva
# CSI 999
# @date: Sep 26, 2021

# Remove any old or system version of Driver, CUDA Toolkit, cuDNN
# sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*"
# sudo apt-get remove --purge nvidia\*
# sudo apt autoremove

# Install NVIDIA driver
# Find latest driver for your video card 
# sudo apt install nvidia-utils-470

# Install CUDA Toolkit
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda
# TEST
# nvcc -V

# Install cuDNN
#
# sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb





export PATH='/usr/local/cuda-11.2.2/bin:$PATH';
echo "Export path done"
export LD_LIBRARY_PATH=/usr/local/cuda-11.2.2/lib64;/usr/local/cuda-11.2.2/include;
echo "Load Library Path set 11.2.2"