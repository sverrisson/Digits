import numpy as np
import csv
import matplotlib.pyplot as plt

# uv venv
# source .venv/bin/activate
# uv pip install numpy matplotlib
# uv run Digits.py

class Digits:
    def show_image(self, image: list):
        a = image
        plt.imshow(a)
        plt.axis('off')  # Turn off axis labels
        plt.show()

    def get_digit(self, digit: np.array):
        _, index = np.unique(digit, return_index=True)
        return index[-1]

    def __init__(self) -> None:
        with open('semeion.data', mode='r') as file:
            data = csv.reader(file, delimiter=' ')
            for row in data:
                image = np.array(row[:256], dtype=float).reshape(16,16)
                digit = np.array(row[256:-1])
                if data.line_num < 3:
                    print(image)
                    print(digit)
                    print(self.get_digit(digit))
                    self.show_image(image)
        

if __name__ == "__main__":
    d = Digits()
    