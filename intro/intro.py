import torch

print(f"Using torch version {torch.__version__}")
# Machine learning uses numbers to propagate loss and do the calculations and all that.
# When we convert a non-numerical data into a numerical data, we call this object a tensor.

# Tensors can have a shape. A scalar is just a 0 dimension tensor.
scalar = torch.tensor(7) # scalars and vectors should be lower-cased
print(f"Scalar has dimension {scalar.ndim} with value {scalar.item()}") # item() only works on scalars

vector = torch.tensor([1,2,3])
print(f"Scalar has dimension {vector.ndim} with shape {vector.shape}")

TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
print(f"Tensor has dimension {TENSOR.ndim} with shape {TENSOR.shape}")

# Usually you start training by defining a shape a filling it up with random numbers to train.
# With more experience, you can define the starting numbers to be closer to the end-product.

random_tensor = torch.rand(size=(3,4))
print(f"Random tensor:\n{random_tensor}\nIt has data type {random_tensor.dtype}")

# You can also use zeros() and ones(). 

zero_to_ten = torch.arange(start=0, end=10, step=1)
ten_zeros = torch.zeros_like(input=zero_to_ten)
ten_ones = torch.ones_like(input=zero_to_ten)

print(zero_to_ten, ten_zeros, ten_ones)

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

# We can do * - + / @ on tensors
# @ is matrix multiplication
TENSOR = (TENSOR + 10 - 5) * 10
print(TENSOR)

# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# torch.matmul(tensor_A, tensor_B) (this will error)
transposed_B = torch.transpose(input=tensor_B, dim0=0, dim1=1)
print(torch.mm(tensor_A, transposed_B))

# Here are some aggregate functions
print(f"Minimum: {zero_to_ten.min()} at position {zero_to_ten.argmin()}")
print(f"Maximum: {zero_to_ten.max()} at position {zero_to_ten.argmax()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {zero_to_ten.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {zero_to_ten.sum()}")




