import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph


# export graph for production
def export_model(sess, saver, input_node_names, output_node_names, graph_name, directory = 'out'):

    tf.keras.backend.set_learning_phase(0)

    # GRAPH SAVING - '.pbtxt'
    tf.train.write_graph(sess.graph.as_graph_def(), directory, graph_name + '_graph.pbtxt')

    # GRAPH SAVING - '.chkp'
    # KEY: This method saves the graph at it's last checkpoint (hence '.chkp')
    saver.save(sess, f'{directory}/' + graph_name + '.chkp')

    # GRAPH SAVING - '.bytes'
    # freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
    # input_binary, checkpoint_path, output_node_names,
    # restore_op_name, filename_tensor_name,
    # output_frozen_graph_name, clear_devices, "")
    freeze_graph.freeze_graph(f'{directory}/' + graph_name + '_graph.pbtxt',
                              None,                 # saver
                              False,
                              f'{directory}/' + graph_name + '.chkp',
                              ",".join(output_node_names),
                              "save/restore_all",
                              "save/Const:0",
                              f'{directory}/frozen_' + graph_name + '.bytes',
                              True,
                              "")

    # GRAPH OPTIMIZING
    input_graph_def = tf.GraphDef()

    with tf.gfile.Open(f'{directory}/frozen_' + graph_name + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

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

    with tf.gfile.FastGFile(f'{directory}/opt_' + graph_name + '.bytes', "wb") as f:
        f.write(constant_graph.SerializeToString())

    print(f"graph saved to {directory}")


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

