import numpy as np
import csv
import matplotlib.pyplot as plt

# uv venv
# source .venv/bin/activate
# uv pip install numpy matplotlib
# uv run Digits.py

line = 0
with open('semeion.data', mode='r') as file:
    data_reader = csv.reader(file, delimiter=' ')
    for row in data_reader:
        image = np.array(row[:256]).reshape(16,16)
        digit = np.array(row[256:-1])
        if line < 1:
            print(image)
            print(digit)
        line += 1


def show_image(image: list):
    a = image
    plt.imshow(a)
    plt.axis('off')  # Turn off axis labels
    plt.show()