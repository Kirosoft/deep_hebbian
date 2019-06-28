import tensorflow as tf
import numpy as np

from .layer import Layer
from .recruitment_cluster import RecruitmentCluster


class RecruitmentLayer(Layer):
    """
    Represents the spatial pooling computation layer
    """
    def __init__(self, output_dim, nodes_per_cluster= 25, num_clusters=6,
                 sparsity=0.02, lr=1e-2, pool_density=0.9,
                 duty_cycle=1000, boost_strength=100, **kwargs):
        """
        Args:
            - output_dim: Size of the output dimension
            - sparsity: The target sparsity to achieve
            - lr: The learning rate in which permenance is updated
            - pool_density: Percent of input a cell is connected to on average.
        """
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.lr = lr   # learning rate multiplier
        self.pool_density = pool_density
        self.duty_cycle = duty_cycle
        self.boost_strength = boost_strength
        self.top_k = int(np.ceil(self.sparsity * np.prod(self.output_dim)))

        # create
        self.recruitment_clusters = [ RecruitmentCluster(nodes_per_cluster, i, sparsity, lr, pool_density,
                                        duty_cycle, boost_strength) for i in range(num_clusters)]

        super().__init__(**kwargs)

    def build(self, input_shape):

        for cluster in self.recruitment_clusters:
            cluster.build(input_shape)

        self.cluster_activation = tf.Variable(tf.zeros([1, self.output_dim]),name="cluster_activation_matrix")

        super().build(input_shape)

    def call(self, x):

        output = tf.stack([self.recruitment_clusters[i].call(x) for i in range(len(self.recruitment_clusters))],name="output_stack")

        return output

    def train(self, x, y):
        """
            x -- cluster input
            y -- cluster output
        """

        layer_out = []

        for cluster in self.recruitment_clusters:
            cluster_out =  cluster.train(x, y)
            layer_out.append(cluster_out)

        return layer_out
