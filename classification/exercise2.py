from torch import nn
import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Linear(in_features=2, out_features=5)
		self.layer2 = nn.Linear(in_features=5, out_features=10)
		self.layer3 = nn.Linear(in_features=10, out_features=1)

	def forward(self, x):
		return self.layer3(self.layer2(self.layer1(x)))


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

if __name__ == '__main__':
	main()