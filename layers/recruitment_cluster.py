'''
Author: Mark Norman
Date: 22nd March 2019
'''
import tensorflow as tf
import numpy as np
from .layer import Layer


class RecruitmentCluster(Layer):
    """
    Agglomerative clustering of large sparse binary vectors using Hebbian style learning rule
    """
    def __init__(self, output_dim, cluster_num, sparsity=0.02, lr=1e-2, pool_density=0.9,
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
        self.cluster_name = str(cluster_num)
        self.cluster_num = cluster_num
        self.activation_threshold = 20

        super().__init__(**kwargs)

    def build(self, input_shape):

        # Permanence of connections between neurons
        self.p = tf.Variable(tf.random_uniform((input_shape[1], self.output_dim), 0, 1), name='Permanence'+self.cluster_name)

        # Connection matrix, dependent on the permenance values
        # If permenance > 0.5, we are connected.
        self.connection = tf.round(self.p, name="connections"+self.cluster_name) #* pool_mask

        # activation level for the cluster
        self.avg_activation = tf.Variable(tf.zeros([1, self.output_dim]),name="activation_matrix"+self.cluster_name)

        super().build(input_shape)

    def call(self, x):

        with tf.name_scope(f"call_cluster_{self.cluster_num}"):
            # Compute the overlap score between input
            overlap = tf.matmul(x, self.connection, name='compute_overlap'+self.cluster_name) #* boost_factor
            batch_size = tf.shape(x, name='batch_size' + self.cluster_name)[0]
            activate = tf.ones((batch_size,overlap.shape[1]))
            inactive = tf.zeros((batch_size,overlap.shape[1]))

            # each node above self.activation_threshold is considered activated i.e. == 1
            output = tf.where(overlap > self.activation_threshold, activate, inactive, name="cluster_out"+self.cluster_name)

        return output

    def train(self, x, layer_y):
        """
        Weight update using Hebbian learning rule.
        Connections are clipped between 0 and 1.
        """

        with tf.name_scope(f"train_cluster_{self.cluster_num}"):

            # the layer_y includes the output from each cluster in the layer
            y = layer_y[self.cluster_num]

            # find the largest cluster activation (for each row in the batch)
            # sum the activated nodes for each cluster and row within the batch
            totals_per_cluster = tf.reduce_sum(layer_y, axis=2)

            totals_per_cluster = tf.reshape(totals_per_cluster,[-1])

            # find the top activated cluster
            vals, idxs = tf.nn.top_k(totals_per_cluster, 1)
            cluster_order = tf.reshape(idxs, [-1])
            top_cluster_index_num = tf.cast(cluster_order[0], tf.int32)

            # Shift input X from 0, 1 to -1, 1.
            x_shifted = 2 * x - 1
            x_zero = x * 0

            # will only learn if this is the top activating cluster otherwise we set the input(x) to zero (x_zero)
            x_shifted = tf.cond(tf.equal(self.cluster_num, top_cluster_index_num), lambda: x_shifted, lambda: x_zero)  #

            batch_size = tf.to_float(tf.shape(x)[0],name='batch_size'+self.cluster_name)

            with tf.name_scope(f"train_cluster_delta_{self.cluster_num}"):

                # multiply the shifted input by the connection matrix and the layer output to get the delta matrix
                delta = tf.einsum('ij,ik,jk->jk', x_shifted, y, self.connection, name="Delta_Matrix"+self.cluster_name) / batch_size

                # Apply learning rate multiplier to the delta matrix
                new_p = tf.clip_by_value(self.p + self.lr * delta, 0, 1)

                # Create train op
                train_op = tf.assign(self.p, new_p, name='train_op_'+self.cluster_name)

        return [train_op, y, self.connection]
