'''
https://github.com/calclavia/htm-tensorflow/blob/master/layers/spatial_pooler.py
'''
import tensorflow as tf
import numpy as np

from .layer import Layer


class SpatialPooler(Layer):
    """
    Represents the spatial pooling computation layer
    """
    def __init__(self, output_dim, sparsity=0.02, lr=1e-2, pool_density=0.9,
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
        super().__init__(**kwargs)

    def build(self, input_shape):

        # Permanence of connections between neurons
        self.p = tf.Variable(tf.random_uniform((input_shape[1], self.output_dim), 0, 1), name='Permanence')

        # Potential pool matrix
        # Masks out the connections randomly
        rand_mask = np.random.binomial(1, self.pool_density, input_shape[1] * self.output_dim)
        pool_mask = tf.constant(np.reshape(rand_mask, [input_shape[1], self.output_dim]), dtype=tf.float32, name='pool_mask')

        # Connection matrix, dependent on the permenance values
        # If permenance > 0.5, we are connected.
        self.connection = tf.round(self.p,name="connections") * pool_mask

        # Time-averaged activation level for each mini-column
        self.avg_activation = tf.Variable(tf.zeros([1, self.output_dim]),name="activation_matrix")

        super().build(input_shape)

    def call(self, x):

        # Boosting calculations
        # The recent activity in the mini-column's (global) neighborhood
        neighbor_mask = tf.constant(-np.identity(self.output_dim) + 1, dtype=tf.float32, name='neighbour_mask')
        neighbor_activity = tf.matmul(self.avg_activation, neighbor_mask, name='neighbour_activity') / (self.output_dim - 1)
        boost_factor = tf.exp(-self.boost_strength * (self.avg_activation - neighbor_activity), name='boost_factor')

        # Compute the overlap score between input
        overlap = tf.matmul(x, self.connection, name='compute_overlap') * boost_factor

        # Compute active mini-columns.
        # The top k activations of given sparsity activates
        batch_size = tf.shape(x,name='batch_size')[0]
        _, act_indicies = tf.nn.top_k(overlap, k=self.top_k, sorted=False, name='top_k')

        # Create a matrix of repeated batch IDs
        batch_ids = tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, self.top_k], name='batch_ids')

        # Stack the batch IDs to generate 2D indices of activated units
        act_indicies = tf.to_int64(tf.reshape(tf.stack([batch_ids, act_indicies], axis=2), [-1, 2]), name='act_indicies')
        act_vals = tf.ones((batch_size * self.top_k,),name='act_vals')
        output_shape = tf.to_int64(tf.shape(overlap),name='output_shape')

        activation = tf.SparseTensor(act_indicies, act_vals, output_shape)
        output = tf.sparse_tensor_to_dense(activation, validate_indices=False,name='output')
        return output

    def train(self, x, y):
        """
        Weight update using Hebbian learning rule.

        For each active SP mini-column, we reinforce active input connections
        by increasing the permanence, and punish inactive connections by
        decreasing the permanence.
        We only want to modify permanences of connections in active mini-columns.
        Ignoring all non-connections.
        Connections are clipped between 0 and 1.
        """

        # Shift input X from 0, 1 to -1, 1.
        x_shifted = 2 * x - 1

        # Compute delta matrix, which contains -1 for all connections to punish
        # and 1 for all connections to reinforce. Use broadcasting behavior.
        batch_size = tf.to_float(tf.shape(x)[0],name='batch_size')

        # active_cons = y * self.connection
        # delta = tf.transpose(x_shifted * tf.transpose(active_cons))
        delta = tf.einsum('ij,ik,jk->jk', x_shifted, y, self.connection, name="Delta_Matrix") / batch_size

        # Apply learning rate multiplier
        new_p = tf.clip_by_value(self.p + self.lr * delta, 0, 1)

        # Create train op
        train_op = tf.assign(self.p, new_p)

        # Update the average activation levels
        avg_activation = tf.reduce_mean(y, axis=0, keep_dims=True)
        new_act_avg = ((self.duty_cycle - 1) * self.avg_activation + avg_activation) / self.duty_cycle
        update_act_op = tf.assign(self.avg_activation, new_act_avg)

        return [train_op, update_act_op, self.connection]
