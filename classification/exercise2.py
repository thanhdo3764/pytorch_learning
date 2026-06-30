from torch import nn
import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
	print("helper_functions.py already exists, skipping download")
else:
	print("Downloading helper_functions.py")
	request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
	with open("helper_functions.py", "wb") as f:
		f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Linear(in_features=2, out_features=5)
		self.layer2 = nn.Linear(in_features=5, out_features=10)
		self.layer3 = nn.Linear(in_features=10, out_features=1)
		self.relu = nn.ReLU()

	def forward(self, x):
		return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))


def accuracy_fn(y_true, y_pred):
	correct = torch.eq(y_true, y_pred).sum().item()
	acc = correct / len(y_pred) * 100
	return acc

def train(
	epochs=100,
	model=None,
	X_train=None,
	y_train=None,
	X_test=None,
	y_test=None,
	loss_fn=None,
	optimizer=None,
	accuracy_fn=None):

	for epoch in range(epochs):
		model.train()
		y_logits = model(X_train).squeeze()
		y_pred = torch.round(torch.sigmoid(y_logits))
		loss = loss_fn(y_logits, y_train)
		acc = accuracy_fn(y_train, y_pred)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		model.eval()
		if epoch % 10 == 0:
			with torch.inference_mode():
				test_logits = model(X_test).squeeze()
				test_pred = torch.round(torch.sigmoid(test_logits))
				test_loss = loss_fn(test_logits, y_test)
				test_acc = accuracy_fn(y_test, test_pred)
				print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
def main():

	device = ""
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	else:
		device = "cpu"

	# Create dataset
	n_samples = 1000
	X, y = make_moons(n_samples, noise=0.03, random_state=42)
	# X has shape [[x0, x1], [x0, x1], [x0, x1]...] and these are NumPy Arrays
	# y is either 0 or 1

	plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap=plt.cm.RdYlBu)
	# plt.show()

	X = torch.from_numpy(X).type(torch.float)
	y = torch.from_numpy(y).type(torch.float)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	my_model = MyModel().to(device)
	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.SGD(params=my_model.parameters(), lr=0.1)

	train(
		epochs=1000,
		model=my_model,
		X_train=X_train,
		y_train=y_train,
		X_test=X_test,
		y_test=y_test,
		loss_fn=loss_fn,
		optimizer=optimizer,
		accuracy_fn=accuracy_fn)

	# Plot decision boundaries for training and test sets
	plt.figure(figsize=(12, 6))
	plt.subplot(1, 2, 1)
	plt.title("Train")
	plot_decision_boundary(my_model, X_train, y_train)
	plt.subplot(1, 2, 2)
	plt.title("Test")
	plot_decision_boundary(my_model, X_test, y_test)
	plt.show()


if __name__ == '__main__':
	main()