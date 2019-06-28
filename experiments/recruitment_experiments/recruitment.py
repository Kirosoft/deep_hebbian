"""
MNIST Example.
Make sure the MNIST dataset is in the data/ folder
"""

import numpy as np
import tensorflow as tf
from math import *

import json
import matplotlib.pyplot as plt
import matplotlib as mpl

# Import the library

epochs = 100
GRAPH_NAME = "deep_hebbian"

def nearest_square(num):
    return round(sqrt(float(num)) + 0.5)

num_classes = 10
num_pixels = 784
pixel_bits = 4
validation_split = 0.9
input_units = nearest_square(125)**2 #num_pixels * pixel_bits
htm_units = 2025
batch_size = 32
#clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_code = None
CLUSTER_DIST = 0.26

train_mode = False

# model
x = []
tf.enable_eager_execution()
cluster_size = 25
num_clusters = 9
cluster_permanences = []
cluster_mask = []
input_size = 2025
pool_density = 0.9
connections = []
sparsity=0.02
top_k = int(np.ceil(sparsity * np.prod(cluster_size)))
lr=1e-2
duty_cycle=1000
avg_activation = []



TrainMode = True

def build():

    # initialise
    for f in range(num_clusters-1):
        cp = tf.random_uniform((cluster_size, input_size), 0, 1)
        cluster_permanences.append(cp)
        rm = np.random.binomial(1, pool_density, input_size * cluster_size)
        cm = np.reshape(rm, [cluster_size, input_size])
        cluster_mask.append(cm)
        connections.append(tf.round(cp)*cm)
        # Time-averaged activation level for each mini-column
        avg_activation.append(tf.zeros([1, cluster_size]))


def train(x):

    outputs = call(x)

    # get the top activating cluster
    order = np.argsort(np.sum(outputs, axis=1))[::-1]

     #adapt the most active clusters
    for f in order[:1]:
        input = np.zeros((1,len(x[0])), dtype=np.float32)
        input[0] = x[0]
        output = np.zeros((1,len(outputs[0])), dtype=np.float32)
        output[0] = outputs[f]
        # Shift input X from 0, 1 to -1, 1.
        x_shifted_input = 2 * input - 1
        batch_size = tf.to_float(tf.shape(input)[0])

        delta = tf.einsum('ij,ik,kj->kj', tf.convert_to_tensor(x_shifted_input), tf.convert_to_tensor(output), tf.convert_to_tensor(np.array(cluster_permanences[f]))) #/ batch_size

        new_p = tf.clip_by_value(cluster_permanences[f] + lr * delta, 0, 1)

        # Create train op
        cluster_permanences[f] =  new_p

        # Update the average activation levels
        #activation = tf.reduce_mean(outputs[f], axis=0, keep_dims=True)
        #new_act_avg = ((duty_cycle - 1) * avg_activation[f] + activation) / duty_cycle
        #avg_activation[f]= new_act_avg
        #print("ok")

    return outputs


def call(x):
    outputs = np.zeros((len(cluster_permanences),cluster_size), dtype=np.float32)

    # loop through the connection for each cluster
    for f in range(len(cluster_permanences)):
        cp = cluster_permanences[f]
        overlap = np.zeros((cluster_size),dtype=np.float32)
        d = 0
        for y in cp:
            a = np.dot(x[0], np.round(y))
            #print(a)
            overlap[d]=a if a > 20 else 0
            d += 1
        batch_size = tf.shape(x)[0]
        #_, act_indicies = tf.nn.top_k(overlap, k=top_k, sorted=False, name='top_k')
        act_indicies_1d = [i for i in np.argsort(overlap)[::-1] if overlap[i] >= 20]
        act_indicies_2d = [[0,i] for i in act_indicies_1d]

        # Create a matrix of repeated batch IDs
        #batch_ids = tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, top_k])

        # Stack the batch IDs to generate 2D indices of activated units
        #act_indicies = tf.to_int64(tf.reshape(tf.stack([batch_ids, act_indicies], axis=2), [-1, 2]), name='act_indicies')
        if (len(act_indicies_1d) > 0):
            act_vals = tf.to_int64(tf.ones(len(act_indicies_1d)))
            output_shape = tf.zeros([1,cluster_size],dtype=tf.float32)

            activation = tf.SparseTensor(act_indicies_2d, act_vals, output_shape.shape)
            # TODO: Keeping it as a sparse tensor is more efficient.
            a = tf.sparse_tensor_to_dense(activation, validate_indices=False)
            outputs[f] = a[0]

    return outputs


def plot_output(input_data, outputs, h1, h2):
    count = 0
    root = nearest_square(input_units)

    array_stack = outputs[0].reshape(5,5)
    for f in range(1, len(outputs)):
        a = outputs[f].reshape(5,5)
        array_stack = np.hstack((array_stack, a))


    #data = output[count].reshape(16, 16)
    h1.set_data(np.array(input_data).reshape(45,45))
    h2.set_data(array_stack)
    #h2.set_clim(0, 50)
    plt.draw()
    plt.pause(.00001)
    count += 1

def main():
    lines=[]
    input_data = np.zeros(shape=(45,45))
    output_data = np.zeros(shape=(16,16))

    fig = plt.figure(figsize=(2, 2))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    plt.ion()

    # define the colors
    cmap1 = mpl.colors.ListedColormap(['w', 'k'])

    # create a normalize object the describes the limits of
    # each color
    bounds = [0., 0.5, 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)

    root = nearest_square(input_units)
    h1 = ax1.imshow(input_data, extent=(0, 45, 45, 0), interpolation='nearest', cmap=cmap1, norm=norm)
    h2 = ax2.imshow(output_data, extent=(0, 35, 5, 0), interpolation='nearest', cmap=cmap1, norm=norm)
    #h3 = ax2.imshow(output_data, extent=(0, 35, 5, 0), interpolation='nearest', cmap=cmap1, norm=norm)

    with open("out/output_sdrs.txt", "r") as f:
        lines = f.readlines()
        f.close()

    with open("out/input_set.json", "r") as f:
        input_set= json.load(f)
        f.close()

    for line in lines:
        x.append([float(v) for v in line.split(',')])

    build()

    if TrainMode:
        for epoch in range(3):
            print("Epoch Train: ", epoch)
            for sample in x:
                v = [sample]
                outputs = train(v)
                plot_output(sample, outputs, h1,h2)


    for epoch in range(1):
        count = 0
        anon=0
        print("Epoch Run: ", epoch)
        for sample in x:
            v = [sample]
            outputs = call(v)
            plot_output(sample, outputs, h1,h2)
            totals = np.sum(outputs, axis=1)
            grand_total = np.sum(outputs)
            sorted = [i for i in np.argsort(totals)[::-1] if totals[i] > 0]
            input = input_set[count]
            if (len(sorted) == 0 or totals[sorted[0]]<=1):
                num = 0 if len(sorted) == 0 else totals[sorted[0]]
                anon+=1
                print(f"Anon {anon/count} - ("+str(num)+","+str(grand_total)+"): ",input)
            count += 1

    print(f"Anon {anon/count}%")
    print("End")


if __name__ == '__main__':
    main()




