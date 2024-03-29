#!/bin/bash

# Installation script used for VNN-COMP. The tool is only compatible with Ubuntu 22.04.

TOOL_NAME=alpha-beta-CROWN
VERSION_STRING=v1
if [[ -z "${VNNCOMP_PYTHON_PATH}" ]]; then
	VNNCOMP_PYTHON_PATH=/home/ubuntu/miniconda/envs/alpha-beta-crown/bin
fi

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing $TOOL_NAME"
TOOL_DIR=$(dirname $(dirname $(realpath $0)))

export DEBIAN_FRONTEND=noninteractive
sudo -E DEBIAN_FRONTEND=noninteractive apt purge -y snapd unattended-upgrades
sudo killall -9 unattended-upgrade-shutdown
sudo -E DEBIAN_FRONTEND=noninteractive apt update
sudo -E DEBIAN_FRONTEND=noninteractive apt upgrade -y
sudo -E DEBIAN_FRONTEND=noninteractive apt install -y sudo vim-gtk curl wget git cmake tmux aria2 build-essential netcat expect dkms aria2

grep AMD /proc/cpuinfo > /dev/null && echo "export MKL_DEBUG_CPU_TYPE=5" >> ${HOME}/.profile
echo "export OMP_NUM_THREADS=1" >> ${HOME}/.profile

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p ${HOME}/miniconda
echo 'export PATH=${PATH}:'${HOME}'/miniconda/bin' >> ~/.profile
echo "alias py37=\"source activate alpha-beta-crown\"" >> ${HOME}/.profile
export PATH=${PATH}:$HOME/miniconda/bin

# Install NVIDIA driver
aria2c -x 10 -s 10 -k 1M "https://us.download.nvidia.com/tesla/515.48.07/NVIDIA-Linux-x86_64-515.48.07.run"
sudo nvidia-smi -pm 0
chmod +x ./NVIDIA-Linux-x86_64-515.48.07.run
sudo ./NVIDIA-Linux-x86_64-515.48.07.run --silent --dkms
# Remove old driver (if already installed) and reload the new one.
sudo rmmod nvidia_uvm; sudo rmmod nvidia_drm; sudo rmmod nvidia_modeset; sudo rmmod nvidia
sudo modprobe nvidia; sudo nvidia-smi -e 0; sudo nvidia-smi -r -i 0
sudo nvidia-smi -pm 1
# Make sure GPU shows up.
nvidia-smi

# Install conda environment
${HOME}/miniconda/bin/conda env create --name alpha-beta-crown -f ${TOOL_DIR}/complete_verifier/environment_vnncomp22.yml
${VNNCOMP_PYTHON_PATH}/pip install -U --no-deps git+https://github.com/dlshriver/DNNV.git@4d4b124bd739b4ddc8c68fed1af3f85b90386155#egg=dnnv

# Install CPLEX
aria2c -x 10 -s 10 -k 1M "http://d.huan-zhang.com/storage/programs/cplex_studio2210.linux_x86_64.bin"
chmod +x cplex_studio2210.linux_x86_64.bin
cat > response.txt <<EOF
INSTALLER_UI=silent
LICENSE_ACCEPTED=true
EOF
sudo ./cplex_studio2210.linux_x86_64.bin -f response.txt

# Build CPLEX interface
make -C ${TOOL_DIR}/complete_verifier/CPLEX_cuts/

echo "Checking python requirements (it might take a while...)"
if [ "$(${VNNCOMP_PYTHON_PATH}/python -c 'import torch; print(torch.__version__)')" != '1.11.0' ]; then
    echo "Unsupported PyTorch version"
    echo "Installation Failure!"
    exit 1
fi

# Run grbprobe for activating gurobi later.
${VNNCOMP_PYTHON_PATH}/grbprobe
