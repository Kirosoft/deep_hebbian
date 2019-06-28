from layers import RecruitmentLayer
from layers import SpatialPooler
import tensorflow as tf


class DeepHebbianModel:
    def __init__(self, input_units, sp_units, clusters):

        self.input_units = input_units
        self.sp_units=sp_units
        self.clusters=clusters

        self.pooler = SpatialPooler(sp_units, lr=1e-2)
        # Model input
        self.x = tf.placeholder(tf.float32, [None, input_units])
        self.y = self.pooler(self.x)
        self.train_ops = self.pooler.train_ops

        cluster = RecruitmentLayer(sp_units, num_clusters=clusters)
        self.x1 = tf.placeholder(tf.float32, [1, sp_units])
        self.y1 = cluster(self.x1)
        self.recruitment_train_ops = cluster.train_ops
