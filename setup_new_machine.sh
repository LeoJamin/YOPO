#!/bin/bash
# =============================================================
# YOPO Workspace Setup Script for New Machine
# Source machine: Ubuntu 20.04, RTX 2060, CUDA 11.8
# Prerequisites: Ubuntu 20.04 + ROS Noetic installed
# =============================================================
set -e

echo "=== YOPO New Machine Setup ==="

# --- 1. Install Miniconda (if not installed) ---
if ! command -v conda &> /dev/null; then
    echo "[1/6] Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    rm /tmp/miniconda.sh
    echo "Miniconda installed. Please restart your shell, then re-run this script."
    exit 0
else
    echo "[1/6] Miniconda already installed, skipping."
    eval "$(conda shell.bash hook)"
fi

# --- 2. Create workspace structure ---
echo "[2/6] Setting up catkin workspace..."
YOPO_WS="$HOME/yopo_ws"
mkdir -p "$YOPO_WS/src"

# --- 3. Clone the repo ---
echo "[3/6] Cloning YOPO repository..."
if [ ! -d "$YOPO_WS/src/YOPO/.git" ]; then
    cd "$YOPO_WS/src"
    git clone git@github.com:LeoJamin/YOPO.git
    cd YOPO
    git checkout jamine/payload-dev
else
    echo "YOPO repo already exists, pulling latest..."
    cd "$YOPO_WS/src/YOPO"
    git pull
fi

# --- 4. Create conda environment ---
echo "[4/6] Creating conda environment 'yopo'..."
if conda env list | grep -q "^yopo "; then
    echo "Conda env 'yopo' already exists, skipping creation."
else
    # Try yml first, fall back to manual creation + pip
    if conda env create -f "$YOPO_WS/src/YOPO/yopo_env.yml" 2>/dev/null; then
        echo "Created from yopo_env.yml"
    else
        echo "yml failed (likely cross-platform issue), creating manually..."
        conda create -n yopo python=3.8 -y
        conda activate yopo
        # Install PyTorch with CUDA 11.8
        conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        # Install remaining pip packages
        pip install -r "$YOPO_WS/src/YOPO/requirements.txt"
    fi
fi

# --- 5. Build ROS workspace ---
echo "[5/6] Building catkin workspace..."
source /opt/ros/noetic/setup.bash
cd "$YOPO_WS"

# Create top-level CMakeLists.txt symlink if missing
if [ ! -f "$YOPO_WS/src/CMakeLists.txt" ]; then
    ln -s /opt/ros/noetic/share/catkin/cmake/toplevel.cmake "$YOPO_WS/src/CMakeLists.txt"
fi

catkin_make -j$(nproc)
source "$YOPO_WS/devel/setup.bash"

# --- 6. Verify ---
echo "[6/6] Verifying installation..."
conda activate yopo
python -c "
import torch
print(f'Python OK')
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "REMAINING MANUAL STEPS:"
echo "  1. Transfer large files from old machine:"
echo "     rsync -avz oldmachine:~/yopo_ws/src/YOPO/YOPO/saved/ ~/yopo_ws/src/YOPO/YOPO/saved/"
echo "     rsync -avz oldmachine:~/yopo_ws/src/YOPO/dataset/   ~/yopo_ws/src/YOPO/dataset/"
echo ""
echo "  2. Add to your ~/.bashrc:"
echo "     source /opt/ros/noetic/setup.bash"
echo "     source ~/yopo_ws/devel/setup.bash"
echo "     conda activate yopo"
echo ""
echo "  3. Verify ROS:"
echo "     roscore  # in one terminal"
echo "     rostopic list  # in another"
