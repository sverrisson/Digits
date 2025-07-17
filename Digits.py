import numpy as np
import csv
import matplotlib.pyplot as plt

line = 0
with open('semeion.data', mode='r') as file:
    data_reader = csv.reader(file)
    for row in data_reader:
        if line < 1:
            print(row)
        line += 1


# uv venv
# source .venv/bin/activate
# uv pip install numpy matplotlib
# uv run Digits.py
# uv pip install yfinance --upgrade
# ulimit -n 4096


def show_image(image: list):
    a = image
    plt.imshow(a)
    plt.axis('off')  # Turn off axis labels
    plt.show()