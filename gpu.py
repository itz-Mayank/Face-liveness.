import torch

# Function to get the available device
def get_device():
    # Check if CUDA (NVIDIA GPU) is available
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

# Example usage
device = get_device()

# Now you can use `device` in your code to run models on the selected device
# Example:
# model.to(device)
