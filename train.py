import numpy as np
import tensorflow as tf
from tqdm import tqdm
from math import *
from pykafka import KafkaClient
from pykafka.common import OffsetType
from itertools import islice
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from layers.deep_hebbian_model import DeepHebbianModel
from deep_hebbian_utils.inputmodel import InputModel
from deep_hebbian_utils.tensorflow_graph_tools import load_graph
from deep_hebbian_utils.message_process_utils import is_valid_message, pre_process_message
from datetime import datetime
import Geohash
from deep_hebbian_utils.tensorflow_graph_tools import export_model

epochs = 10
nearest_square = lambda num: round(sqrt(float(num)) + 0.5)
input_units: int = nearest_square(125) ** 2
sp_units = 2025
clusters = 6
batch_size = 300
train_mode = False
kafka_url = "kafka:9092"
input_topic = "INPUT_TOPIC"
output_topic = 'OUTPUT_TOPIC'
anomaly_threshold = 2
GRAPH_NAME = "deep_hebbian"
MODEL_DIRECTORY = 'production_model_v1'
encoder_model_file = "./encoder_model_v2.json"


def plot_output(input_data, output, cluster, h1,    h2, h3):
    root = nearest_square(input_units)

    for i in range(len(input_data)):
        inp = input_data[i]
        out = output[i]

        h1.set_data(np.array(inp).reshape(root, root))
        h2.set_data(out.reshape(45, 45))
        cluster_data = np.hstack([c.reshape(5, 5) for c in cluster[i]])
        h3.set_data(cluster_data)

        h2.set_clim(0, 50)
        plt.draw()
        plt.pause(.00001)


def train_spatial_pooler(sess, input_set, model):
    outputs = []
    print('Training spatial pooler...')

    # Shuffle input
    order = np.random.permutation(len(input_set))

    for i in tqdm(range(0, len(order) + 1 - batch_size, batch_size)):
        # Mini-batch training
        batch_indices = order[i:i + batch_size]
        x = [input_set[ii] for ii in batch_indices]
        output = sess.run(model.train_ops, feed_dict={model.x: x})
        outputs.append(output)

    return outputs


def train_recruitment_layer(sess, input_set, model):
    outputs = []
    print('Training Recruitment Layer...')

    # Shuffle input
    order = np.random.permutation(len(input_set))

    for i in range(len(order)):
        output_layer = []
        for train_ops in model.recruitment_train_ops:
            current = [input_set[order[i]]]
            sess.run([train_ops], feed_dict={model.x1: current})

        outputs.append(output_layer)

    return


def get_inference(sess, input_set, h1, h2, h3):
    x = sess.graph.get_tensor_by_name('prefix/Placeholder:0')
    y = sess.graph.get_tensor_by_name('prefix/output:0')
    y1 = sess.graph.get_tensor_by_name('prefix/output_stack:0')
    x1 = sess.graph.get_tensor_by_name('prefix/Placeholder_1:0')

    all_outputs = sess.run(y, feed_dict={x: input_set})
    cluster_output = [sess.run(y1, feed_dict={x1: [output]}) for output in all_outputs]

    #plot_output(input_set, all_outputs, cluster_output, h1, h2, h3)

    return cluster_output


def get_inference_training(sess, input_set, model, h1, h2, h3):
    all_outputs = sess.run(model.y, feed_dict={model.x: input_set})
    cluster_output = [sess.run(model.y1, feed_dict={model.x1: [output]}) for output in all_outputs]

    #plot_output(input_set, all_outputs, cluster_output, h1, h2, h3)

    return cluster_output


def process_data(sess, all_data_conv, all_data, h1, h2, h3):
    cluster_output = get_inference(sess, all_data_conv, h1, h2, h3)
    anon_count = 0

    for f in range(len(all_data)):
        current_data = all_data[f]
        score = 0

        current_cluster_output = cluster_output[f]
        cluster_activity = np.sum(current_cluster_output, axis=2).flatten()
        if np.amax(cluster_activity) < anomaly_threshold:
            score = 1
            anon_count += 1
            print(f'Anon: {(anon_count/len(all_data))*100}/{np.amax(cluster_activity)}: {current_data}')

        event_ts = datetime.utcfromtimestamp(all_data[f]["W_START"] / 1000)
        all_data[f]["SCORE"] = score
        all_data[f]["EVENT_TS"] = f'{event_ts:%Y-%m-%dT%H:%M:%S%z}Z'
        key_vals = all_data[f]["UNIQUEKEY"].split("*")
        all_data[f]["EMAIL"] = key_vals[1]
        all_data[f]["ORG_ID"] = key_vals[0]

        # cleanup some of the more detailed location data
        geo_list = []
        geo_hash = ""
        geo_point = ""
        if "GEO" in all_data[f]:
            for ip in all_data[f]["GEO"]:

                if all_data[f]["GEO"][ip]['latitude'] and all_data[f]["GEO"][ip]['longitude']:
                    print(f"Latitude: {all_data[f]['GEO'][ip]['latitude']}:{all_data[f]['GEO'][ip]['longitude']}")
                    geo_hash = Geohash.encode(all_data[f]["GEO"][ip]['latitude'], all_data[f]["GEO"][ip]['longitude'])
                    geo_point = f'{all_data[f]["GEO"][ip]["latitude"]},{all_data[f]["GEO"][ip]["longitude"]}'
                    geo_obj = {"key": all_data[f]["GEO"][ip]['region_code'],
                               "geo_hash": geo_hash,
                               "location": geo_point,
                               "latitude": all_data[f]["GEO"][ip]['latitude'],
                               "longitude": all_data[f]["GEO"][ip]['longitude'],
                               "name": all_data[f]["GEO"][ip]['region_name']
                               }
                    geo_list.append(geo_obj)

        all_data[f]["GEO"] = geo_list
        all_data[f]["geo_hash"] = geo_hash  # will only store the last location
        all_data[f]["location"] = geo_point  # will only store the last location
        all_data[f]["GEO_NUM_LOCATION"] = len(all_data[f]["GEO"])

        # remove blank addresses
        if "CLIENTIP_HIST" in all_data[f]:
            del all_data[f]["CLIENTIP_HIST"]

        if "REMOTEIP_HIST" in all_data[f]:
            del all_data[f]["REMOTEIP_HIST"]

    return all_data


def main():
    client = KafkaClient(hosts=kafka_url)
    session_topic = client.topics[input_topic]
    session_out_topic = client.topics[output_topic]

    consumer = session_topic.get_simple_consumer(
                    auto_offset_reset=OffsetType.EARLIEST,
                    reset_offset_on_start=False
                )

    # Build the complete model - input layer(encoder), middle layer (spatial pooling), output layer (recruitment clusters)
    model = DeepHebbianModel(input_units, sp_units, clusters)

    sp_size = nearest_square(sp_units)
    data = np.zeros(shape=(sp_size, sp_size))
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

    print('Loading data...')
    input_model = InputModel(encoder_model_file)

    all_data = []
    all_data_conv = []

    for message in islice(consumer, 376048, 710752):

        print(message.offset, message.value)
        if message.value is not None:
            input_data = json.loads(message.value.decode('utf-8'))

            if is_valid_message(input_data):
                input_data = pre_process_message(input_data)
                all_data.append(input_data)
                bit_array = np.array([int(i) for i in input_model.get_model_params(input_data)[0]])
                bit_array = np.hstack([bit_array, np.zeros(input_units - len(bit_array))])
                all_data_conv.append(bit_array)

    print('Processing data...')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

        if not train_mode:
            graph = load_graph(f'{MODEL_DIRECTORY}/frozen_{GRAPH_NAME}.bytes')
            # We can verify that we can access the list of operations in the graph
            for op in graph.get_operations():
                print(op.name)

            sess = tf.Session(graph=graph)

            all_data = process_data(sess, all_data_conv, all_data, h1, h2, h3)

            # Output the new scored date to the target output topic
            with session_out_topic.get_sync_producer() as producer:
                count = 0
                for data in all_data:
                    producer.produce(json.dumps(data).encode())
                    count += 1

        else:
            # Run the 'init' op
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                print('=== Epoch ' + str(epoch) + ' ===')
                train_spatial_pooler(sess, all_data_conv, model)

                all_outputs = sess.run(model.y, feed_dict={model.x: all_data_conv})

            train_recruitment_layer(sess, all_outputs, model)
            get_inference_training(sess, all_data_conv, model, h1, h2, h3)

            saver.save(sess, f"./{MODEL_DIRECTORY}/{GRAPH_NAME}-model", global_step=epochs)

            export_model(sess, saver,
                         ["Placeholder", "Placeholder_1", "Placeholder_2", "Placeholder_3", "Placeholder_4",
                          "Placeholder_5", "Placeholder_6"],
                         ["output", "output_stack", "call_cluster_0/cluster_out0", "call_cluster_1/cluster_out1",
                          "call_cluster_2/cluster_out2", "call_cluster_3/cluster_out3",
                          "call_cluster_4/cluster_out4", "call_cluster_5/cluster_out5"]
                         , GRAPH_NAME, MODEL_DIRECTORY)

    plt.ioff()
    plt.show()

    print("Finished")


if __name__ == '__main__':
    main()
