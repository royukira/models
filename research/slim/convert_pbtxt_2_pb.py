from google.protobuf import text_format
import tensorflow as tf

model_path = input("Input model(.pbtxt) path: ")
graph_dir = input("Output dir path: ")

with open(model_path, 'r') as f:
    text_graph = f.read()
graph_def = text_format.Parse(text_graph, tf.GraphDef())
tf.train.write_graph(graph_def, graph_dir, 'graph.pb', as_text=False)