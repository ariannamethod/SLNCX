#!/bin/bash

# Function to install dependencies for Debian/Ubuntu
install_debian_dependencies() {
    echo "Installing dependencies for Debian-based systems..."
    sudo apt update
    sudo apt install -y git python3 python3-pip transmission-cli
}

# Function to install dependencies for RPM-based systems
install_rpm_dependencies() {
    echo "Installing dependencies for RPM-based systems..."
    sudo dnf install -y git python3 python3-pip transmission-cli
}

# Function to install dependencies for macOS
install_macos_dependencies() {
    echo "Installing dependencies for macOS..."
    # Install Homebrew if not installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install git python transmission-cli
}

# Function to check if a command is installed
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to install required Python packages
install_python_packages() {
    echo "Installing required Python packages..."
    pip3 install --upgrade pip
    pip3 install huggingface_hub[hf_transfer] || { echo "Failed to install huggingface_hub."; exit 1; }
    pip3 install -r requirements.txt || { echo "Failed to install requirements."; exit 1; }
}

# Detect the OS and install appropriate dependencies
echo "Detecting OS..."
OS="$(uname)"
if [[ "$OS" == "Linux" ]]; then
    echo "Detected Linux OS."
    # Check if the system is Debian-based
    if command_exists apt; then
        install_debian_dependencies
    # Check if the system is RPM-based
    elif command_exists dnf; then
        install_rpm_dependencies
    else
        echo "Unsupported Linux distribution. Please install git, python3, python3-pip, and transmission-cli manually."
        exit 1
    fi
elif [[ "$OS" == "Darwin" ]]; then
    install_macos_dependencies
else
    echo "Detected Windows or WSL. Checking for Git and Python..."
    # Check if Git is installed
    if ! command_exists git; then
        echo "Git is not installed. Please install Git from https://git-scm.com/"
        exit 1
    fi

    # Check if Python is installed
    if ! command_exists python; then
        echo "Python is not installed. Please install Python from https://www.python.org/downloads/"
        exit 1
    fi

    # Install required Python packages
    install_python_packages
    exit 0
fi

# Clone the repository and navigate into it
echo "Cloning the repository..."
git clone https://github.com/xai-org/grok-1.git && cd grok-1 || { echo "Failed to clone the repository."; exit 1; }

# Install required Python packages
install_python_packages

# Download model weights from Hugging Face
echo "Downloading model weights from Hugging Face..."
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False || { echo "Failed to download weights from Hugging Face."; exit 1; }

# Function to download weights using the torrent link
download_weights() {
    echo "Downloading weights using the torrent link..."
    transmission-cli "magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce" || { echo "Failed to download weights using torrent."; exit 1; }
}

# Call the function to download weights
download_weights

# Run the application
echo "Running the application..."
python3 run.py || { echo "Failed to run the application."; exit 1; }

echo "Script completed successfully."
