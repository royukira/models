import tensorflow as tf

model_path = "/roy_work/Object_detection_API/pre_trained_model/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18/frozen_inference_graph.pb" #input("Input model(.pb) path: ")
graph_dir = "/roy_work/Object_detection_API/pre_trained_model/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18/vis_pb/" #input("Output dir path: ")

graph = tf.Graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.io.gfile.GFile(model_path, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter(graph_dir, graph)

print("Graph is written to {}".format(graph_dir))