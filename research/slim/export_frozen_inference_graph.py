from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.tools import freeze_graph

from datasets import dataset_factory
from nets import nets_factory

slim = contrib_slim

tf.app.flags.DEFINE_string(
    'trained_checkpoint_prefix', None,
    'Path to trained checkpoint, typically of the form '
    'path/to/model.ckpt')

tf.app.flags.DEFINE_integer(
    'batch_size', 1,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v2', 'The name of the architecture to save.')

tf.app.flags.DEFINE_integer(
    'image_size', 224,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'pci_HeadHat_dav4_cls',
    'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string('output_directory', None, 'Path to write outputs.')

# tf.app.flags.DEFINE_bool(
#     'quantize', False, 'whether to use quantized graph or not.')

# tf.app.flags.DEFINE_bool(
#     'is_video_model', False, 'whether to use 5-D inputs for video model.')

# tf.app.flags.DEFINE_integer(
#     'num_frames', None,
#     'The number of frames to use. Only used if is_video_model is True.')

tf.app.flags.DEFINE_bool(
    'write_text_graphdef', False,
    'Whether to write a text version of graphdef.')

tf.app.flags.DEFINE_boolean(
    'write_inference_graph', False,
    'If true, writes inference graph to disk.')

tf.app.flags.DEFINE_bool(
    'use_grayscale', False,
    'Whether to convert input images to grayscale.')

FLAGS = tf.app.flags.FLAGS

def build_cls_graph(batch_size,
                    image_size, 
                    network_fn,
                    use_grayscale=False,
                    graph=tf.get_default_graph()):
    """Build the classification graph."""
    with graph.as_default() as default_graph:
        if use_grayscale:
            input_shape = (batch_size, image_size, image_size, 1)
        else:
            input_shape = (batch_size, image_size, image_size, 3)
        input_placeholder = tf.placeholder(
            tf.uint8, input_shape, "input_tensor"
        )
        input_tensor = tf.cast(input_placeholder, tf.float32)
        _, end_points = network_fn(input_tensor)

        slim.get_or_create_global_step()

        """把output的node从endpoints提出来，重新命名，加进去default graph，因为endpoints字典的
        键一般和graph图上的节点名字不一样，例如：
             endpoints       tf.get_default_grpah().as_default_graph()
            'Prediction' == 'MobilenetV2/Prediction/Softmax'
        """
        outputs = add_output_tensor_nodes(end_points)

        return default_graph, input_placeholder, outputs

def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
    """Writes the graph and the checkpoint into disk."""
    for node in inference_graph_def.node:
        node.device = ''
    with tf.Graph().as_default():
        tf.import_graph_def(inference_graph_def, name='')
        with tf.Session() as sess:
            saver = tf.train.Saver(
                saver_def=input_saver_def, save_relative_paths=True)
            print(">> filename_tensor_name: {}".format(saver.saver_def.filename_tensor_name))
            print(">> restore_op_name: {}".format(saver.saver_def.restore_op_name))
            input("Continue...")
            saver.restore(sess, trained_checkpoint_prefix)
            saver.save(sess, model_path)

def write_saved_model(saved_model_path,
                      frozen_graph_def,
                      inputs,
                      outputs):
    """Writes SavedModel to disk.

    If checkpoint_path is not None bakes the weights into the graph thereby
    eliminating the need of checkpoint files during inference. If the model
    was trained with moving averages, setting use_moving_averages to true
    restores the moving averages, otherwise the original set of variables
    is restored.

    Args:
        saved_model_path: Path to write SavedModel.
        frozen_graph_def: tf.GraphDef holding frozen graph.
        inputs: The input placeholder tensor.
        outputs: A tensor dictionary containing the outputs of a DetectionModel.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:

            tf.import_graph_def(frozen_graph_def, name='')

            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

            tensor_info_inputs = {
                'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

            detection_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=tensor_info_inputs,
                    outputs=tensor_info_outputs,
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))

            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants
                    .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        detection_signature,
                },
            )
            builder.save()

def add_output_tensor_nodes(endpoints_tensors,
                            output_collection_name='inference_op'):
    """Adds output nodes for detection boxes and scores.

    Adds the following nodes for output tensors -
        * classification_scores: float32 tensor of shape [batch_size, num_classes]
        containing scores for images.
    Args:
        endpoints_tensors: a dictionary from components of the network to the corresponding activation tensor
        'classification_scores': [batch, num_classes]
        output_collection_name: Name of collection to add output tensors to.

    Returns:
        A tensor dict containing the added output tensor nodes.
    """
    endpoint_node_name = 'Predictions'
    output_node_name = 'classification_scores'
    scores = endpoints_tensors[endpoint_node_name]
    outputs = {}
    outputs[output_node_name] = tf.identity(
        scores, name=output_node_name)

    for output_key in outputs:
        tf.add_to_collection(output_collection_name, outputs[output_key])

    return outputs

def _export_inference_graph(model_name,
                            image_size,
                            batch_size,
                            grayscale,
                            dataset,
                            labels_offset,
                            trained_checkpoint_prefix,
                            output_directory,
                            write_inference_graph,
                            write_text_graphdef):
    
    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory,
                                   '{}_frozen_inference_graph.pb'.format(model_name))
    saved_model_dir = os.path.join(output_directory, "{}_saved_model".format(model_name))
    save_model_path = os.path.join(output_directory, 'model.ckpt')

    num_classes = dataset_factory.get_dataset_num_class(dataset)
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=(num_classes - labels_offset),
        is_training=False
    )

    graph, input_placeholder, outputs = build_cls_graph(
        batch_size, image_size, network_fn, grayscale)
    
    """注意：这顺序不能调换，saver要放在graph_def前面，因为调用saver时候，会向graph添加了"save/xxx"节点
    所以要加完了，才调用 as_graph_def()，提早调用了会导致丢失save节点 """
    saver = tf.train.Saver()
    input_saver_def = saver.as_saver_def()
    graph_def = graph.as_graph_def()
    
    write_graph_and_checkpoint(
        graph_def,
        save_model_path,
        input_saver_def,
        trained_checkpoint_prefix)
    
    if write_text_graphdef:
        txt_output_file = os.path.join(
            output_directory, 
            '{}_text_graph.txt'.format(model_name))
        tf.io.write_graph(
          graph_def,
          output_directory,
          txt_output_file,
          as_text=True)
    
    if write_inference_graph:
        inference_graph_path = os.path.join(output_directory,
                                        'inference_graph.pbtxt')
        for node in graph_def.node:
            node.device = ''
        with tf.gfile.GFile(inference_graph_path, 'wb') as f:
            f.write(str(graph_def))
    
    output_node_names = ",".join(outputs.keys())
    tensor_name_list = [tensor.name for tensor in graph_def.node]
    print(tensor_name_list)
    input("Continue...")

    frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
      input_graph_def=graph_def,
      input_saver_def=input_saver_def,
      input_checkpoint=trained_checkpoint_prefix,
      output_node_names=output_node_names,
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      output_graph=frozen_graph_path,
      clear_devices=True,
      initializer_nodes='')

    write_saved_model(
        saved_model_dir,
        frozen_graph_def,
        input_placeholder,
        outputs)
    
def export_inference_graph(model_name,
                           image_size,
                           batch_size,
                           grayscale,
                           dataset,
                           labels_offset,
                           trained_checkpoint_prefix,
                           output_directory,
                           write_inference_graph,
                           write_text_graphdef):
    _export_inference_graph(
        model_name, image_size, batch_size, grayscale,
        dataset, labels_offset, trained_checkpoint_prefix,
        output_directory, write_inference_graph, write_text_graphdef
    )

def main(_):
    if not FLAGS.output_directory:
        raise ValueError('You must supply the DIR path to save to with --output_directory')
    if not FLAGS.trained_checkpoint_prefix:
        raise ValueError('You must supply the DIR path to save to with --trained_checkpoint_prefix')
    tf.logging.set_verbosity(tf.logging.INFO)
    export_inference_graph(
        FLAGS.model_name,
        FLAGS.image_size,
        FLAGS.batch_size,
        FLAGS.use_grayscale,
        FLAGS.dataset_name,
        FLAGS.labels_offset,
        FLAGS.trained_checkpoint_prefix,
        FLAGS.output_directory,
        FLAGS.write_inference_graph,
        FLAGS.write_text_graphdef
    )

if __name__ == "__main__":
    tf.app.run()