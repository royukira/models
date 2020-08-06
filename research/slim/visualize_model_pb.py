# Visualize the .pb model file via Tensorboard

import tensorflow as tf

model_path = input("Input model(.pb) path: ") #"/roy_work/Object_detection_API/pre_trained_model/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18/frozen_inference_graph.pb" #input("Input model(.pb) path: ")
graph_dir = input("Output dir path: ") # "/roy_work/Object_detection_API/pre_trained_model/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18/vis_pb" #input("Output dir path: ")

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.io.gfile.GFile(model_path, 'rb').read())
print("Importing graph...")
tf.import_graph_def(graph_def, name='graph')
print("Imported!")
summaryWriter = tf.summary.FileWriter(graph_dir, graph)
print("Graph is written to {}".format(graph_dir))