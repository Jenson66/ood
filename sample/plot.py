#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle


def make_plot(xs: list, labels: list, filename: str) -> None:
    '''Make a boxplot from a list of data points'''
    plt.figure(figsize=(10, 7))
    for x, label in zip(xs, labels):
        plt.plot(x, label=label)
    plt.xlabel("Epochs")
    plt.ylabel('Rate')
    plt.legend()
    plt.savefig(filename, dpi=320)
    print(f"Saved the plot as {filename}")


if __name__ == "__main__":
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)
    make_plot(list(results.values()), list(results.keys()), "plot.png")