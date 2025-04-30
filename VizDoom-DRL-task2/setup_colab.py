import os
import sys
import subprocess
import torch

def setup_colab():
    """Setup ViZDoom and dependencies for Google Colab"""
    print("Setting up ViZDoom environment for Colab...")
    
    # Check if we're in Colab
    try:
        from google.colab import drive
        IN_COLAB = True
        print("Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("Not running in Google Colab")
    
    if IN_COLAB:
        # Mount Google Drive
        drive.mount('/content/gdrive')
        
        # Create necessary directories
        os.makedirs('/content/gdrive/MyDrive/ML4_vizdoom', exist_ok=True)
        os.makedirs('/content/gdrive/MyDrive/ML4_vizdoom/models', exist_ok=True)
        os.makedirs('/content/gdrive/MyDrive/ML4_vizdoom/logs', exist_ok=True)
        os.makedirs('/content/gdrive/MyDrive/ML4_vizdoom/videos', exist_ok=True)
        
        # Install dependencies
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vizdoom"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "stable-baselines3"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("No GPU available")
            
        # Set environment variables for ViZDoom
        os.environ['VK_ICD_FILENAMES'] = '/usr/share/vulkan/icd.d/nvidia_icd.json'
        os.environ['VK_LAYER_PATH'] = '/usr/share/vulkan/explicit_layer.d'
        
        print("Setup complete!")
    else:
        print("This script is meant to be run in Google Colab")

if __name__ == "__main__":
    setup_colab() 