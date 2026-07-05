import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

class MyModel(nn.Module):
	def __init__(self, input_features, output_features, hidden_units=8):
		super().__init__()
		self.linear_layer_stack = nn.Sequential(
			nn.Linear(in_features=input_features, out_features=hidden_units),
			nn.ReLU(),
			nn.Linear(in_features=hidden_units, out_features=hidden_units),
			# nn.ReLU(),
			nn.Linear(in_features=hidden_units, out_features=hidden_units),
			nn.ReLU(),
			nn.Linear(in_features=hidden_units, out_features=output_features))

	def forward(self, x):
		return self.linear_layer_stack(x)


def create_multiclass_data():
	import numpy as np
	N = 100 # number of points per class
	D = 2 # dimensionality
	K = 3 # number of classes
	X = np.zeros((N*K,D)) # data matrix (each row = single example)
	y = np.zeros(N*K, dtype='uint8') # class labels
	for j in range(K):
		ix = range(N*j,N*(j+1))
		r = np.linspace(0.0,1,N) # radius
		t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
		X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
		y[ix] = j
		# lets visualize the data
	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	# plt.show()

	return train_test_split(
		torch.from_numpy(X).type(torch.float),
		torch.from_numpy(y).type(torch.LongTensor),
		test_size=0.2,
		random_state=RANDOM_SEED
	)

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
		y_logits = model(X_train)
		y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
		loss = loss_fn(y_logits, y_train)
		acc = accuracy_fn(y_train, y_pred)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		model.eval()
		if epoch % 10 == 0:
			with torch.inference_mode():
				test_logits = model(X_test)
				test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
				test_loss = loss_fn(test_logits, y_test)
				test_acc = accuracy_fn(y_test, test_pred)
				print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


def get_device():
	if torch.cuda.is_available():
		return "cuda"
	elif torch.backends.mps.is_available():
		return "mps"
	else:
		return "cpu"

def main():
	device = get_device()
	X_train, X_test, y_train, y_test = create_multiclass_data()
	my_model = MyModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES).to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(params=my_model.parameters(), lr=0.01)

	train(
		epochs=2000,
		model=my_model,
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
		loss_fn=loss_fn,
		optimizer=optimizer,
		accuracy_fn=accuracy_fn)

	my_model.eval()
	with torch.inference_mode():
		logits = my_model(X_test)
		preds = logits.argmax(dim=1)

	print(f"Predictions: {preds[:10]}\nLabels: {y_test[:10]}")
	print(f"Test accuracy: {accuracy_fn(y_true=y_test, y_pred=preds)}%")

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