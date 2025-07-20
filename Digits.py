import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_map

# uv venv
# source .venv/bin/activate
# uv pip install -U numpy matplotlib mlx scikit-learn
# uv run Digits.py

# A simple loss function.
def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

# compute the accuracy of the model on the validation set
def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

# Neural network model class
class MLP(nn.Module):
    # NN model setup
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()

        # Set up layers
        self.layers = [
            nn.Linear(in_dims, 128),
            nn.Linear(128, 128),
            nn.Linear(128, out_dims)
        ]

    # Computation implementation
    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = mx.maximum(x, 0.0) if i > 0 else x
            x = l(x)
        return x

class Digits:
    features = []  # Features
    labels = []  # Labels

    # Show image for the 16 x 16 array
    def show_image(self, image: list):
        a = image.reshape(16, 16)
        plt.imshow(a)
        plt.axis('off')  # Turn off axis labels
        plt.show()

    # Get the digit that is set in the training data
    def get_digit(self, digit: np.array):
        _, index = np.unique(digit, return_index=True)
        return index[-1]
    
    # Split the data for training and testing
    def split_for_testing(self, test_size: float):
        # 1593 records, train 1274, test 319
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size,
            # random_state=42,     # For reproducibility
            shuffle=True,        # Shuffle before splitting
            stratify=None        # Set to y if you want stratified sampling
        )

        print(self.X_train.shape)
        # self.show_image(self.X_train[-1])
        # print(self.get_digit(self.y_train[-1]))
        # print('')
        print(self.X_test.shape)
        # self.show_image(self.X_test[-1])
        # print(self.get_digit(self.y_test[-1]))


    def __init__(self, samples: int = 1593, test_size: float = 0.20) -> None:
        # 1593 records, train 1274, test 319
        with open('semeion.data', mode='r') as file:
            data = csv.reader(file, delimiter=' ')
            for row in data:
                image = row[:256]
                digit = row[256:-1]
                self.features.append(image)
                self.labels.append(digit)
                # print(self.get_digit(digit))
                # self.show_image(image)

        # Finally convert to numpy
        self.X = np.array(self.features, dtype=np.float32)
        self.y = np.array(self.labels, dtype=np.float32)
        
        # Split the data for training and testing
        self.split_for_testing(test_size)
        

if __name__ == "__main__":
    d = Digits()

    mlp = MLP(256, 10)
    print(mlp)
    params = mlp.parameters()
    shapes = tree_map(lambda p: p.shape, mlp.parameters())
    print(shapes)
    print(params["layers"][0]["weight"].shape) # (128, 256)
    print(params["layers"][0])

    loss_and_grad_fn = nn.value_and_grad(mlp, loss_fn)
    optimizer = opt.Adam(learning_rate=0.001)
    X_train = mx.array(d.X_train)
    y_train = mx.array(d.y_train)
    print(f"Training samples: {X_train.shape}")

    n_epochs = 60
    for epoch in range(n_epochs):
        loss, grads = loss_and_grad_fn(mlp, X_train, y_train)
        mlp.update(optimizer.apply_gradients(grads, mlp))
        mx.eval(mlp.parameters(), optimizer.state)
        if epoch % 10 == 0:
            print(f"Loss after {epoch} steps: {loss.item():.4f}")
        if loss.item() < 0.004:
            print(f"Final Loss after {epoch} steps: {loss.item():.4f}")
            break
    