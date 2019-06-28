"""
MNIST Example.
Make sure the MNIST dataset is in the data/ folder
"""

import numpy as np
import tensorflow as tf
import random
import sys
from math import *

from deep_hebbian_utils.inputmodel import InputModel
from layers import SpatialPooler
import string
from pykafka import KafkaClient
from pykafka.common import OffsetType
from itertools import islice
import json
from scipy.spatial import distance
# Import the library
from deep_hebbian_utils.onehot import OneHot
import pickle
from datetime import datetime

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

train_mode = True


class HTMModel:
    def __init__(self):
        pooler = SpatialPooler(htm_units, lr=1e-2)
        # Model input
        self.x = tf.placeholder(tf.float32, [None, input_units])
        self.y = pooler(self.x)
        self.train_ops = pooler.train_ops

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


def get_sdrs(sess, input_set, model,h1,h2):
    sdrs = []
    print('Get sdrs')

    x = sess.graph.get_tensor_by_name('prefix/Placeholder:0')
    y = sess.graph.get_tensor_by_name('prefix/output:0')

    all_outputs = sess.run(y, feed_dict={ x: input_set })

    for x in all_outputs:
        matched, min_dist = find_closest_sdr(sdrs,x)

        # exact match
        if min_dist == 0 :
            matched['count'] += 1
        # similar - merge vectors
        elif min_dist <= CLUSTER_DIST:
            matched['count'] += 1
            found = False
            for hist in matched['history']:
                if np.array_equal(x.astype(np.bool),hist):
                    found = True
                    break
            if not found:
                matched['history'].append(x.astype(np.bool))
                matched['merged'] += 1

            matched['data'] = np.logical_or(matched['data'], x.astype(np.bool))
        else:
            # different - create new entry
            sdrs.append({'id':len(sdrs),'data': x.astype(np.bool), 'merged': 0, 'count': 1, 'history': [x.astype(np.bool)]})

        check_balance(sdrs)

    process_sdr_stats(sdrs)
    display_sdrs(sdrs)

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


def process_batch(all_data, all_data_conv, model, sdrs):
    print('Processing data...')

    input_set = np.array(all_data_conv)

    graph = load_graph('out/frozen_deep_hebbian.bytes')

    with tf.Session(graph = graph) as sess:

        x = sess.graph.get_tensor_by_name('prefix/Placeholder:0')
        y = sess.graph.get_tensor_by_name('prefix/output:0')

        all_outputs = sess.run(y, feed_dict={x: input_set})

        for f in range(len(all_outputs)):
            output = all_outputs[f]
            clusters, min_dist, avg_dist = find_closest_sdrs(sdrs, output)

            score = 1
            if len(clusters) < 1:
                score = 0.2
            elif len(clusters) < 2:
                score  = 0.5
            elif len(clusters) < 3:
                score  = 0.8

            event_ts = datetime.utcfromtimestamp(all_data[f]["W_START"] / 1000)
            all_data[f]["SCORE"] = 1-score
            all_data[f]["EVENT_TS"] = f'{event_ts:%Y-%m-%dT%H:%M:%S%z}Z'
            key_vals = all_data[f]["UNIQUEKEY"].split("*")
            all_data[f]["EMAIL"]=key_vals[1]
            all_data[f]["ORD_ID"]=key_vals[0]

            if len(clusters) < 2:
                print("anom: ",all_data[f])

    return all_data


def main():
    global client_code
    global h1
    client_code = id_generator()
    global all_data
    global num_data
    global all_data_conv
    sdrs = []
    #clientsocket.connect(('localhost', 13000))
    #cmd = '{"command":"start","client":"'+client_code+'"}&'
    #clientsocket.send(cmd.encode("utf-8"))

    with open("out/sdr_clusters.json", "rb") as f:
        sdrs = pickle.load(f)

    client = KafkaClient(hosts="my-kafka-cluster:9092")
    session_topic = client.topics['SESSION_GEO']
    session_out_topic = client.topics['SESSION_OUT']

    consumer = session_topic.get_simple_consumer(
        auto_offset_reset=OffsetType.EARLIEST,
        reset_offset_on_start=False
    )

    # Build a model
    model = HTMModel()

    print('Loading data...')
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

    scored_data = process_batch(all_data, all_data_conv, model, sdrs)

    with session_out_topic.get_sync_producer() as producer:
        count=0
        for data in scored_data:
            producer.produce(json.dumps(data).encode())
            count += 1


if __name__ == '__main__':
    main()
