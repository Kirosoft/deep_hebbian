"""
SimpSOM (Simple Self-Organizing Maps) v1.3.3
F. Comitani @2017 
 
A lightweight python library for Kohonen Self-Organising Maps (SOM).
"""

from __future__ import print_function

import numpy as np

from .somnet import SOMNet
from .somnode import SOMNode


def run_colorsExample():
    """Example of usage of SimpSOM: a number of vectors of length three
        (corresponding to the RGB values of a color) are used to briefly train a small network.
        Different example graphs are then printed from the trained network.
    """

    raw_data = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0.2, 0.2, 0.5]])
    labels = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'indigo']

    print(
        "Welcome to SimpSOM (Simple Self Organizing Maps) v1.3.3!\nHere is a quick example of what this library can do.\n")
    print("The algorithm will now try to map the following colors: ", end=' ')
    for i in range(len(labels) - 1):
        print((labels[i] + ", "), end=' ')
    print("and " + labels[-1] + ".\n")

    net = SimpSOM.SOMNet(20, 20, raw_data, PBC=True)

    net.colorEx = True
    net.train(0.01, 10000)

    print("Saving weights and a few graphs...", end=' ')
    net.save('colorExample_weights')
    net.nodes_graph()

    net.diff_graph()
    net.project(raw_data, labels=labels)
    net.cluster(raw_data, type='qthresh')

    print("done!")


if __name__ == "__main__":

	run_colorsExample()
