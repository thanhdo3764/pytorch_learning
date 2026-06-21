import torch

# Working with NVIDIA GPU will speed things up
print(f"Checking for GPU: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# We can check for Apple silicon GPU
print(f"Checking for Apple GPU: {torch.backends.mps.is_available()}")

device = ""

if torch.cuda.is_available():
	device = "cuda"
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"

tensor = torch.tensor([1,2,3])
print(f"Tensor is on device: {tensor.device}")

# Lets try moving it to GPU
tensor_on_gpu = tensor.to(device=device)
print(f"Tensor is now on device: {tensor_on_gpu.device}")

# NumPu only works on CPU, so lets convert it back
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
