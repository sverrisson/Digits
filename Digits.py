import numpy as np
import csv
import matplotlib.pyplot as plt

# uv venv
# source .venv/bin/activate
# uv pip install numpy matplotlib
# uv run Digits.py

def show_image(image: list):
    a = image
    plt.imshow(a)
    plt.axis('off')  # Turn off axis labels
    plt.show()

line = 0
with open('semeion.data', mode='r') as file:
    data_reader = csv.reader(file, delimiter=' ')
    for row in data_reader:
        image = np.array(row[:256], dtype=float).reshape(16,16)
        digit = np.array(row[256:-1])
        if line < 4:
            print(image)
            print(digit)
            show_image(image)
        line += 1
