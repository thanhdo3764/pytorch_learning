import torch
from torch import nn
import matplotlib.pyplot as plt

class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
		self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

	def forward(self, X):
		return self.weights*X + self.bias

def plot_predictions(
	train_data=None, 
	train_labels=None, 
	test_data=None, 
	test_labels=None, 
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

def train(
	epochs=100,
	model=None,
	X_train=None,
	y_train=None,
	X_test=None,
	y_test=None,
	loss_fn=None,
	optimizer=None):

	for epoch in range(epochs):
		model.train()
		y_pred = model(X_train)
		loss = loss_fn(y_pred, y_train)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if epoch % 20 == 0:
			with torch.inference_mode():
				test_pred = model(X_test)
				test_loss = loss_fn(test_pred, y_test.type(torch.float))
				print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

			



def main():
	device = ""
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	else:
		device = "cpu"

	weight = 0.3
	bias = 0.9

	# Create data
	start = 0
	end = 1
	step = 0.02
	X = torch.arange(start, end, step).unsqueeze(dim=1) # [, , ,] -> [[],[],[]]
	y = weight * X + bias

	# Create Train and Test
	split = int(0.8 * len(X))
	X_train, y_train = X[:split], y[:split]
	X_test, y_test = X[split:], y[split:]

	# Plot data
	plot_predictions(
		train_data=X_train,
		train_labels=y_train,
		test_data=X_test, 
		test_labels=y_test)

	# plt.show()

	my_model = MyModel()
	print(my_model.state_dict())

	loss_fn = nn.L1Loss()
	optimizer = torch.optim.SGD(params=my_model.parameters(), lr=0.0001)

	train(
		epochs=100000,
		model=my_model,
		X_train=X_train,
		y_train=y_train,
		X_test=X_test,
		y_test=y_test,
		loss_fn=loss_fn,
		optimizer=optimizer)

	my_model.eval()
	with torch.inference_mode():
		train_preds = my_model(X_train)
		test_preds = my_model(X_test)

		plt.scatter(torch.cat((X_train, X_test), dim=0), torch.cat((train_preds, test_preds), dim=0), c="r", s=4, label="predictions")

	plt.show()
	

if __name__ == '__main__':
	main()