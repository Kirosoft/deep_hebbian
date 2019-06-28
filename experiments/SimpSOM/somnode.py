
from __future__ import print_function

import sys
import numpy as np
from experiments import SimpSOM as hx


class SOMNode:
    """ Single Kohonen SOM Node class. """

    def __init__(self, x, y, numWeights, netHeight, netWidth, PBC, minVal=[], maxVal=[], pcaVec=[], weiArray=[]):

        """Initialise the SOM node.

        Args:
            x (int): Position along the first network dimension.
            y (int): Position along the second network dimension
            numWeights (int): Length of the weights vector.
            netHeight (int): Network height, needed for periodic boundary conditions (PBC)
            netWidth (int): Network width, needed for periodic boundary conditions (PBC)
            PBC (bool): Activate/deactivate periodic boundary conditions.
            minVal(np.array, optional): minimum values for the weights found in the data
            maxVal(np.array, optional): maximum values for the weights found in the data
            pcaVec(np.array, optional): Array containing the two PCA vectors.
            weiArray (np.array, optional): Array containing the weights to give
                to the node if a file was loaded.


        """

        self.PBC = PBC
        self.pos = hx.coorToHex(x, y)
        self.weights = []

        self.netHeight = netHeight
        self.netWidth = netWidth

        if weiArray == [] and pcaVec == []:
            # select randomly in the space spanned by the data
            for i in range(numWeights):
                if round(np.random.random()) >= 0.5:
                    self.weights.append(1)
                else:
                    self.weights.append(0)

            # self.weights.append(np.random.random()*(maxVal[i]-minVal[i])+minVal[i])
        elif weiArray == [] and pcaVec != []:
            # select uniformly in the space spanned by the PCA vectors
            self.weights = (x - self.netWidth / 2) * 2.0 / self.netWidth * pcaVec[0] + (
                        y - self.netHeight / 2) * 2.0 / self.netHeight * pcaVec[1]
        else:
            for i in range(numWeights):
                self.weights.append(weiArray[i])

    def hamming(self, vec1, vec2):

        diff_bits = np.logical_xor(vec1, vec2)
        hamming = np.sum(diff_bits)
        return hamming

    def get_distance_hamming(self, vec):

        """Calculate the distance between the weights vector of the node and a given vector.

        Args:
            vec (np.array): The vector from which the distance is calculated.

        Returns:
            (float): The distance between the two weight vectors.
        """

        sum = 0
        if len(self.weights) == len(vec):
            return self.hamming(self.weights, vec)
        else:
            sys.exit("Error: dimension of nodes != input data dimension!")

    def get_distance(self, vec):

        """Calculate the distance between the weights vector of the node and a given vector.

        Args:
            vec (np.array): The vector from which the distance is calculated.

        Returns:
            (float): The distance between the two weight vectors.
        """

        sum = 0
        if len(self.weights) == len(vec):
            for i in range(len(vec)):
                sum += (self.weights[i] - vec[i]) * (self.weights[i] - vec[i])
            return np.sqrt(sum)
        else:
            sys.exit("Error: dimension of nodes != input data dimension!")

    def get_nodeDistance(self, node):

        """Calculate the distance within the network between the node and another node.

        Args:
            node (somNode): The node from which the distance is calculated.

        Returns:
            (float): The distance between the two nodes.

        """

        if self.PBC == True:

            """ Hexagonal Periodic Boundary Conditions """

            if self.netHeight % 2 == 0:
                offset = 0
            else:
                offset = 0.5

            return np.min([np.sqrt((self.pos[0] - node.pos[0]) * (self.pos[0] - node.pos[0]) \
                                   + (self.pos[1] - node.pos[1]) * (self.pos[1] - node.pos[1])),
                           # right
                           np.sqrt(
                               (self.pos[0] - node.pos[0] + self.netWidth) * (self.pos[0] - node.pos[0] + self.netWidth) \
                               + (self.pos[1] - node.pos[1]) * (self.pos[1] - node.pos[1])),
                           # bottom
                           np.sqrt((self.pos[0] - node.pos[0] + offset) * (self.pos[0] - node.pos[0] + offset) \
                                   + (self.pos[1] - node.pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (
                                               self.pos[1] - node.pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                           # left
                           np.sqrt(
                               (self.pos[0] - node.pos[0] - self.netWidth) * (self.pos[0] - node.pos[0] - self.netWidth) \
                               + (self.pos[1] - node.pos[1]) * (self.pos[1] - node.pos[1])),
                           # top
                           np.sqrt((self.pos[0] - node.pos[0] - offset) * (self.pos[0] - node.pos[0] - offset) \
                                   + (self.pos[1] - node.pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (
                                               self.pos[1] - node.pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                           # bottom right
                           np.sqrt((self.pos[0] - node.pos[0] + self.netWidth + offset) * (
                                       self.pos[0] - node.pos[0] + self.netWidth + offset) \
                                   + (self.pos[1] - node.pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (
                                               self.pos[1] - node.pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                           # bottom left
                           np.sqrt((self.pos[0] - node.pos[0] - self.netWidth + offset) * (
                                       self.pos[0] - node.pos[0] - self.netWidth + offset) \
                                   + (self.pos[1] - node.pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (
                                               self.pos[1] - node.pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                           # top right
                           np.sqrt((self.pos[0] - node.pos[0] + self.netWidth - offset) * (
                                       self.pos[0] - node.pos[0] + self.netWidth - offset) \
                                   + (self.pos[1] - node.pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (
                                               self.pos[1] - node.pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                           # top left
                           np.sqrt((self.pos[0] - node.pos[0] - self.netWidth - offset) * (
                                       self.pos[0] - node.pos[0] - self.netWidth - offset) \
                                   + (self.pos[1] - node.pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (
                                               self.pos[1] - node.pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4))])

        else:
            return np.sqrt((self.pos[0] - node.pos[0]) * (self.pos[0] - node.pos[0]) \
                           + (self.pos[1] - node.pos[1]) * (self.pos[1] - node.pos[1]))

    def update_weights(self, inputVec, sigma, lrate, bmu):

        """Update the node Weights.

        Args:
            inputVec (np.array): A weights vector whose distance drives the direction of the update.
            sigma (float): The updated gaussian sigma.
            lrate (float): The updated learning rate.
            bmu (somNode): The best matching unit.
        """

        dist = self.get_nodeDistance(bmu)
        gauss = np.exp(-dist * dist / (2 * sigma * sigma))
        if gauss > 0:
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - gauss * lrate * (self.weights[i] - inputVec[i])

    def update_bits(self, source_vec, target_vec, num_bits):
        result_vec = np.copy(source_vec)
        diff_bits = np.logical_xor(source_vec, target_vec)
        ix = np.nonzero(diff_bits == 1)
        if (len(ix[0]) > 0):
            choices = np.random.choice(ix[0], min(len(ix[0]), num_bits))

            for idx in choices:
                result_vec[idx] = target_vec[idx]

            return result_vec
        else:
            print("No change, numbits: ", num_bits, "diff bits: ", len(ix))
            return source_vec

    def update_weights_hamming(self, inputVec, sigma, lrate, bmu):

        """Update the node Weights.

        Args:
            inputVec (np.array): A weights vector whose distance drives the direction of the update.
            sigma (float): The updated gaussian sigma.
            lrate (float): The updated learning rate.
            bmu (somNode): The best matching unit.
        """
        MAX_CHANGE_BITS = 8

        dist = self.get_nodeDistance(bmu)
        gauss = np.exp(-dist * dist / (2 * sigma * sigma))
        if gauss > 0 and dist > 0:
            num_bits = int(round((gauss * lrate) + 0.5))
            if (num_bits < 1):
                num_bits = 1
                print("oops no change: " + str(dist), "Gauss: " + str(gauss))
            self.weights = self.update_bits(self.weights, inputVec, num_bits)
        # self.weights[i] = self.weights[i] - gauss * lrate * (self.weights[i] - inputVec[i])