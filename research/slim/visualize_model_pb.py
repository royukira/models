# Visualize the .pb model file via Tensorboard

import tensorflow as tf

model_path = input("Input model path: ")
graph_dir = input("Output dir path: ")

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.io.gfile.GFile(model_path, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter(graph_dir, graph)

print("Graph is written to {}".format(graph_dir))