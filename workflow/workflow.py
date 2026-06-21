import torch
from torch import nn
import matplotlib.pyplot as plt

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(f"X: {X[:10]}\n y: {y[:10]}")

# Lets build a model to learn the X --> y relationship
# You need a training set, testing set, and sometimes a validation set.
# Training is for helping the model learn
# Testing set is used to see if the model has improved
# Validation set gets tuned on this data

# Lets split the training and test 80%:20%
split = int(0.8 * len(X))
X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# This is to visualize
def plot_predictions(
	train_data=X_train, 
	train_labels=y_train, 
	test_data=X_test, 
	test_labels=y_test, 
	predictions=None):
	"""
	Plots training data, test data and compares predictions.
	"""
	plt.figure(figsize=(10, 7))

	# Plot training data in blue
	plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
	
	# Plot test data in green
	plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

	if predictions is not None:
		# Plot the predictions in red (predictions were made on the test data)
		plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

	# Show the legend
	plt.legend(prop={"size": 14});

plot_predictions()
# plt.show()

# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)

torch.manual_seed(42)

model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

# We can use torch.inference_mode() to get predictions

# Make predictions with model
with torch.inference_mode(): 
    y_preds = model_0(X_test)

# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

plot_predictions(predictions=y_preds)

# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))

torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
    	# 1. Forward pass on test data
    	test_pred = model_0(X_test)

    	# 2. Calculate loss on test data
    	test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

    	# Print out what's happening
    	if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 

loaded_model_0 = LinearRegressionModel
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
