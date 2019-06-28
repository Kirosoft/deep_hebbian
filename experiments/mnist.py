"""
MNIST Example.
Make sure the MNIST dataset is in the data/ folder
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tqdm import tqdm
import random
import time
import sys
from math import *

from deep_hebbian_utils.inputmodel import InputModel
from layers import SpatialPooler
import string
from pykafka import KafkaClient
from pykafka.common import OffsetType
from itertools import islice
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance
# Import the library
from experiments.SimpSOM import SOMNet
from deep_hebbian_utils.onehot import OneHot
from tensorflow.python.framework import graph_util
from layers import RecruitmentLayer

epochs = 10
GRAPH_NAME = "deep_hebbian"

def nearest_square(num):
    return round(sqrt(float(num)) + 0.5)

num_classes = 10
num_pixels = 784
pixel_bits = 4
validation_split = 0.9
input_units = nearest_square(125)**2 #num_pixels * pixel_bits
htm_units = 2025
batch_size = 300
recruitment_batch_size=2
#clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_code = None
CLUSTER_DIST = 0.26

train_mode = False


class HTMModel:
    def __init__(self):
        pooler = SpatialPooler(htm_units, lr=1e-2)
        # Model input
        self.x = tf.placeholder(tf.float32, [None, input_units])
        self.y = pooler(self.x)
        self.train_ops = pooler.train_ops

        cluster = RecruitmentLayer(htm_units, num_clusters=6)
        self.x1 = tf.placeholder(tf.float32, [1, htm_units])
        self.y1 = cluster(self.x1)
        self.recruitment_train_ops = cluster.train_ops

        # Build classifier
        #classifier_in = Input((htm_units,))
        #classifier_out = Dense(num_classes, activation='softmax')(classifier_in)
        #self.classifier = Model(classifier_in, classifier_out)
        #self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def convert_bytes_tostring(bytes):
    return bytes.decode('utf-8')


def is_valid_message(message):
    valid_message_codes = [1009,1010,1011,1012,1033,1034,1035,1036,1037,1047]
    result = False

    for msg in message['EVENT_HIST'].keys():
        msgCode = int(msg)
        result = msgCode in valid_message_codes
        if result:
            break

    return result


def pre_process_message(inputData):
    fail_messages = [1033,1034,1035,1036,1037]
    success_messages = [1009,1010,1011,1012,1047]

    # add duration
    end = inputData['W_END']
    start = inputData['W_START']
    inputData['EVENT_DURATION'] = end - start
    inputData['SUCCESS_EVENT_COUNT']= min(9,sum([ inputData['EVENT_HIST'][mc] for mc in inputData['EVENT_HIST'].keys() if int(mc) in success_messages]))
    inputData['FAIL_EVENT_COUNT']= min(9,sum([ inputData['EVENT_HIST'][mc] for mc in inputData['EVENT_HIST'].keys() if int(mc) in fail_messages]))
    inputData['TIME_OF_DAY_ENUM'] = OneHot.get_time_of_day_enum(start)
    return inputData


def validate(sess, val_set, model,h1,h2):
    print('Validating...')

    # Feed into HTM layer
    all_outputs = sess.run(model.y, feed_dict={ model.x: val_set })

    # Feed into classifier layer
    #loss, accuracy = model.classifier.evaluate(np.array(all_outputs), np.array(val_labels))
    #print('Accuracy: {}'.format(accuracy))

    plot_output(val_set, all_outputs,h1,h2)

    time.sleep(.000001)


def plot_output(input_data, output, cluster, h1, h2,h3):

    root = nearest_square(input_units)

    for i in range(len(input_data)):

        inp = input_data[i]
        out = output[i]

        h1.set_data(np.array(inp).reshape(root,root))
        h2.set_data(out.reshape(45, 45))
        cluster_data = np.hstack([c.reshape(5,5) for c in cluster[i]])
        h3.set_data(cluster_data)

        h2.set_clim(0, 50)
        plt.draw()
        plt.pause(.00001)


def train_htm(sess, input_set, model):
    outputs = []
    print('Training HTM...')

    # Train HTM layer
    # Shuffle input
    order = np.random.permutation(len(input_set))

    for i in tqdm(range(0, len(order) + 1 - batch_size, batch_size)):
        # Mini-batch training
        batch_indices = order[i:i+batch_size]
        x = [input_set[ii] for ii in batch_indices]
        output= sess.run(model.train_ops, feed_dict={ model.x: x })
        outputs.append(output)
        #send_training_data(i,output)
        #plt.show(draw=False)

    return outputs

def train_recruitment_layer(sess, input_set, model, merged, epoch):
    outputs = []
    print('Training Recruitment Layer...')

    # Shuffle input
    order = np.random.permutation(len(input_set))

    for i in range(len(order)):
        output_layer = []
        for train_ops in model.recruitment_train_ops:
            current = [ input_set[order[i]] ]
            output = sess.run([train_ops], feed_dict={ model.x1: current })
            #output_layer.append(output)

        #if i % 100:
        #    writer.add_summary(summary, i)

        outputs.append(output_layer)
        #send_training_data(i,output)
        #plt.show(draw=False)

    return outputs


def plot_som(sess, input_set, model, ax3, ax4, val_set, sdrs):

    raw_data = sess.run(model.y, feed_dict={ model.x: input_set })

    # Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions.
    net = SOMNet(10, 10, raw_data, PBC=False) # loadFile='filename_weights',

    # Train the network for 10000 epochs and with initial learning rate of 0.1.
    net.train(500, 1000)

    # Save the weights to file
    net.save('filename_weights')

    # Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
    # and then according to the distance between each node and its neighbours.
    #net.nodes_graph(colnum=0)
    print("Calc diff graph")
    net.diff_graph(show=True, useGraph = ax3)

    # Project the datapoints on the new 2D network map.
    print("Project 2D map")
    #net.project(raw_data)

    print("Calc clusters")
    # Cluster the datapoints according to the Quality Threshold algorithm.
    clusters = net.cluster(raw_data, cutoff=3, type='KMeans',show=True, useGraph=ax4, numcl=6)
    print("Done")

    htm = sess.run(model.y, feed_dict={ model.x: val_set })
    for h in range(0,len(htm)):
        idx = net.find_bmu_idx(htm[h])
        orig = all_data[num_data+h]
        orig_input = all_data_conv[num_data+h]

        for c in range(0,len(clusters)):
            if idx in clusters[c]:
                print("SOM Cluster: "+str(c))

        matching_clusters,min_dist, avg_dist = find_closest_sdrs(sdrs, htm[h])
        result_list = ', '.join([str(x['id']) for x in matching_clusters])
        print("Clusters: "+ result_list)
        if len(matching_clusters) <= 3:
            print("Anomalous: ",str(min_dist), str(avg_dist))
            print(orig)

def get_inference(sess, input_set,h1, h2, h3):
    x = sess.graph.get_tensor_by_name('prefix/Placeholder:0')
    y = sess.graph.get_tensor_by_name('prefix/output:0')
    y1 = sess.graph.get_tensor_by_name('prefix/output_stack:0')
    x1 = sess.graph.get_tensor_by_name('prefix/Placeholder_1:0')

    all_outputs = sess.run(y, feed_dict={ x: input_set })
    cluster_output = [sess.run(y1, feed_dict={ x1: [output]}) for output in all_outputs]

    plot_output(input_set, all_outputs,cluster_output, h1,h2, h3)

    return cluster_output

def get_sdrs(sess, input_set, model,h1, h2, h3):
    sdrs = []
    print('Get sdrs')

    #x = sess.graph.get_tensor_by_name('prefix/Placeholder:0')
    #y = sess.graph.get_tensor_by_name('prefix/output:0')

    all_outputs = sess.run(model.y, feed_dict={ model.x: input_set })
    cluster_output = [sess.run(model.y1, feed_dict={ model.x1: [output]}) for output in all_outputs]

    # with open("out/output_sdrs.txt", "w+") as f:
    #     for v in all_outputs:
    #         out = ','.join([str(x) for x in v])
    #         f.write(out+'\n')
    #     f.flush()
    #     f.close()

    plot_output(input_set, all_outputs,cluster_output, h1,h2, h3)

    # for x in all_outputs:
    #     matched, min_dist = find_closest_sdr(sdrs,x)
    #
    #     # exact match
    #     if min_dist == 0 :
    #         matched['count'] += 1
    #     # similar - merge vectors
    #     elif min_dist <= CLUSTER_DIST:
    #         matched['count'] += 1
    #         found = False
    #         for hist in matched['history']:
    #             if np.array_equal(x.astype(np.bool),hist):
    #                 found = True
    #                 break
    #         if not found:
    #             matched['history'].append(x.astype(np.bool))
    #             matched['merged'] += 1
    #
    #         matched['data'] = np.logical_or(matched['data'], x.astype(np.bool))
    #     else:
    #         # different - create new entry
    #         sdrs.append({'id':len(sdrs),'data': x.astype(np.bool), 'merged': 0, 'count': 1, 'history': [x.astype(np.bool)]})
    #
    #     check_balance(sdrs)
    #
    # process_sdr_stats(sdrs)
    # display_sdrs(sdrs)
    #
    # with open("out/sdr_clusters.json", "wb") as f:
    #     pickle.dump(sdrs, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     f.flush()
    #     f.close()

    return sdrs


def check_balance(sdrs):

    for sdr in sdrs:
        for h in sdr['history']:
            dist = distance.hamming(sdr['data'],h)
            if dist > CLUSTER_DIST:
                print("no longer belongs")


def display_sdrs(sdrs):

    for sdr in sdrs:
        print('---------------------')
        print('idx: ', sdr['id'])
        print('Count: ', sdr['count'])
        print('Merged: ', sdr['merged'])
        print('avg_dist: ',sdr['avg_distance'])
        print('max_dist: ', sdr['max_dist'])
        print('min_dist: ', sdr['min_dist'])
        print('diff_dist: ', sdr['max_dist'] - sdr['min_dist'])


def process_sdr_stats(sdrs):

    for sdr in sdrs:
        avg_distance = 0
        min_dist = sys.maxsize
        max_dist = 0

        for i in sdrs:
            dist = distance.hamming(sdr['data'],i['data'])
            avg_distance += dist
            if dist < min_dist and dist != 0:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

        avg_distance /= len(sdrs)-1
        sdr['avg_distance'] = avg_distance
        sdr['max_dist'] = max_dist
        sdr['min_dist'] = min_dist
        if dist < CLUSTER_DIST:
            print('clusters should merge: ',sdr['id'])


def find_closest_sdr(sdrs, target):
    min_dist = sys.maxsize
    found = None

    for sdr in sdrs:
        dist = distance.hamming(sdr['data'], target)
        if dist < min_dist:
            min_dist = dist
            found = sdr

    return found, min_dist


def find_closest_sdrs(sdrs, target):
    result_list = []
    min_dist = sys.maxsize
    avg_dist = 0

    for sdr in sdrs:
        dist = distance.hamming(sdr['data'], target)
        if dist < CLUSTER_DIST:
            result_list.append(sdr)
            avg_dist += dist
        if dist < min_dist:
            min_dist = dist

    return result_list, min_dist, avg_dist/max(len(result_list),1)


# export graph for Unity
def export_model(sess, saver, input_node_names, output_node_names):
    # # We use a built-in TF helper to export variables to constants
    # output_graph_def = tf.graph_util.convert_variables_to_constants(
    #     sess,  # The session is used to retrieve the weights
    #     tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
    #     output_node_names.split(",")  # The output node names are used to select the usefull nodes
    # )

    tf.keras.backend.set_learning_phase(0)


    # GRAPH SAVING - '.pbtxt'
    tf.train.write_graph(sess.graph.as_graph_def(), 'out', GRAPH_NAME + '_graph.pbtxt')

    # GRAPH SAVING - '.chkp'
    # KEY: This method saves the graph at it's last checkpoint (hence '.chkp')
    saver.save(sess, 'out/' + GRAPH_NAME + '.chkp')

    # GRAPH SAVING - '.bytes'
    # freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
    # input_binary, checkpoint_path, output_node_names,
    # restore_op_name, filename_tensor_name,
    # output_frozen_graph_name, clear_devices, "")
    freeze_graph.freeze_graph('out/' + GRAPH_NAME + '_graph.pbtxt',
                              None,                 # saver
                              False,
                              'out/' + GRAPH_NAME + '.chkp',
                              ",".join(output_node_names),
                              "save/restore_all",
                              "save/Const:0",
                              'out/frozen_' + GRAPH_NAME + '.bytes',
                              True,
                              "")

    # GRAPH OPTIMIZING
    input_graph_def = tf.GraphDef()

    with tf.gfile.Open('out/frozen_' + GRAPH_NAME + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    # output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    #         input_graph_def, input_node_names, [output_node_name],
    #         tf.float32.as_datatype_enum)
    #
    # for fixing the bug of batch norm
    gd = input_graph_def
    for node in gd.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        gd,
        output_node_names)

    with tf.gfile.FastGFile('out/opt_' + GRAPH_NAME + '.bytes', "wb") as f:
        f.write(constant_graph.SerializeToString())

    print("graph saved!")


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def main():
    global client_code
    global h1
    client_code = id_generator()
    global all_data
    global num_data
    global all_data_conv
    global writer

    #clientsocket.connect(('localhost', 13000))
    #cmd = '{"command":"start","client":"'+client_code+'"}&'
    #clientsocket.send(cmd.encode("utf-8"))

    client = KafkaClient(hosts="my-kafka-client:9092")
    session_topic = client.topics['SESSION_TOPIC']

    consumer = session_topic.get_simple_consumer(
        auto_offset_reset=OffsetType.EARLIEST,
        reset_offset_on_start=False
    )

    # Build a model
    model = HTMModel()

    data = np.zeros(shape=(45,45))
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)

    plt.ion()

    # define the colors
    cmap1 = mpl.colors.ListedColormap(['w', 'k'])

    # create a normalize object the describes the limits of
    # each color
    bounds = [0., 0.5, 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)

    root = nearest_square(input_units)
    h1 = ax1.imshow(data, extent=(0, root, root, 0), interpolation='nearest', cmap=cmap1, norm=norm)
    h2 = ax2.imshow(data, extent=(0, 45, 45, 0), interpolation='nearest', cmap=cmap1, norm=norm)
    h3 = ax3.imshow(data, extent=(0, 30, 5, 0), interpolation='nearest', cmap=cmap1, norm=norm)

    # Load MNSIT
    print('Loading data...')
    #mnist = input_data.read_data_sets("data/", one_hot=False)
    inputModel = InputModel("./session_model.json")

    # Process data using simple greyscale encoder
    all_data = []
    all_data_conv = []

    for message in islice(consumer, 0, 10000):

        print(message.offset, message.value)
        if message.value != None:
            inputData = json.loads(message.value.decode('utf-8'))

            if is_valid_message(inputData):
                inputData = pre_process_message(inputData)
                all_data.append(inputData)
                bit_array = np.array([int(i) for i in inputModel.get_model_params(inputData)[0]])
                bit_array = np.hstack([bit_array, np.zeros(input_units-len(bit_array))])
                all_data_conv.append(bit_array)

    print('Processing data...')

    num_data = int(len(all_data_conv) * validation_split)
    num_validate = len(all_data_conv) - num_data

    input_set = np.array(all_data_conv[:num_data])
    #input_labels = all_labels[:num_data]
    val_set = all_data_conv[num_data:num_data+num_validate]
    #val_labels = all_labels[num_data:num_data+num_validate]
    with open("out/input_set.txt", "w+") as f:
        for v in val_set:
            out = ','.join([str(x) for x in v])
            f.write(out+'\n')
        f.flush()
        f.close()

    with open("out/input_set.json", "w+") as f:
        json.dump(all_data, f)
        f.flush()
        f.close()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

        if not train_mode:
            #new_saver = tf.train.import_meta_graph('./htm-model/htm-model-100.meta')
            #new_saver.restore(sess, tf.train.latest_checkpoint('./htm-model/.'))
            graph = load_graph('out/frozen_deep_hebbian.bytes')
            # We can verify that we can access the list of operations in the graph
            for op in graph.get_operations():
                print(op.name)

            sess = tf.Session(graph = graph)

            cluster_output = get_inference(sess, input_set, h1, h2, h3)

        else:
            #sess = tf.InteractiveSession()
            #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "127.0.0.1:8080")
            # Run the 'init' op
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                print('=== Epoch ' + str(epoch) + ' ===')
                output_set= train_htm(sess, input_set, model)

                all_outputs = sess.run(model.y, feed_dict={model.x: input_set})


            cluster_output= train_recruitment_layer(sess, all_outputs, model, merged, epoch)
            sdrs = get_sdrs(sess, input_set, model, h1, h2, h3)

            saver.save(sess, "./htm-model/htm-model",global_step=epochs)
            export_model(sess, saver, ["Placeholder","Placeholder_1","Placeholder_2","Placeholder_3","Placeholder_4","Placeholder_5","Placeholder_6"],
                         ["output","output_stack","call_cluster_0/cluster_out0","call_cluster_1/cluster_out1","call_cluster_2/cluster_out2","call_cluster_3/cluster_out3","call_cluster_4/cluster_out4","call_cluster_5/cluster_out5"])

            # final training to build SDR list


    # np.set_printoptions(linewidth=20000,threshold=np.nan)
    #
    # clusters, idx_list = get_clusters(all_outputs, 20, True, 80)
    # print(idx_list)
    # print(f"Clusters: {len(list(set(idx_list)))}")
    # for idx1, cluster in enumerate(clusters):
    #     if cluster['id'] != -1:
    #         print(f"Cluster: {cluster['id']}")
    #         for h in cluster['history']:
    #             print(h['data'].astype(int))
    #     else:
    #         print(f"Cluster: {idx1} merged with {cluster['merged_with']}")

        #plot_som(sess, val_set, model, ax3, ax4, val_set, sdrs)

        # validate the output
        #validate(sess)
    #clientsocket.close()

    plt.ioff()
    plt.show()

    print("Finished")
    input()


if __name__ == '__main__':
    main()
