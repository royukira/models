# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from losses import loss_factory

# cosine_decay_with_warmup need to be imported from object_detection API
import sys
sys.path.append('/my_github/models/research')
from object_detection.utils import learning_schedules as thirdparty_ls

slim = contrib_slim

MODE_MAP = {
  'train':0,
  'eval':1    
}

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy. Note For '
                            'historical reasons loss from all clones averaged '
                            'out and learning rate decay happen per clone '
                            'epochs')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

# New Params
tf.app.flags.DEFINE_integer(
  'validation_interval_steps', 1000,
  'The frequency with which the model is validated, in steps.'
)

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

tf.app.flags.DEFINE_integer(
    'quantize_delay', -1,
    'Number of steps to start quantized training. Set to -1 would disable '
    'quantized training.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    '"polynomial" or "cosine_decay_with_warmup"(new) ')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.01, 
    'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# New Param
tf.app.flags.DEFINE_float(
    'warmup_learning_rate', 0.01,
    'Initial learning rate for warm up (Only for cosine_with_warmup)'
)

# New Param
tf.app.flags.DEFINE_integer(
    'warmup_steps', 1000,
    'Number of warmup steps. (Only for cosine_with_warmup)'
)

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test/train_eval split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

# New param
tf.app.flags.DEFINE_integer(
    'eval_batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

# New param
tf.app.flags.DEFINE_integer(
    'eval_image_size', 300, 'eval image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', 200000,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_bool('use_grayscale', False,
                         'Whether to convert input images to grayscale.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

#######################
# Loss function Flags #
#######################

tf.app.flags.DEFINE_string(
    'loss_fn', 'softmax_ce',
    'Choose a suitable loss function: softmax_ce, softmax_focal_loss'
)


FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  # Note: when num_clones is > 1, this will actually have each clone to go
  # over each epoch FLAGS.num_epochs_per_decay times. This is different
  # behavior from sync replicas and is expected to produce different results.
  decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                    FLAGS.batch_size)

  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.end_learning_rate,
                                      power=1.0,
                                      cycle=False,
                                      name='polynomial_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'cosine_decay_with_warmup':
    return thirdparty_ls.cosine_decay_with_warmup(global_step,
                                                  FLAGS.learning_rate,
                                                  FLAGS.max_number_of_steps,
                                                  warmup_learning_rate=FLAGS.warmup_learning_rate,
                                                  warmup_steps=FLAGS.warmup_steps)
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                      FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
  return optimizer


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        break
    else:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    # 返回一个列表，包含training set 和 dev set
    dataset_list = []  # 0 - train; 1 - dev
    if FLAGS.dataset_split_name == "train_eval":
      dataset_list.append(dataset_factory.get_dataset(
        FLAGS.dataset_name, 'train', FLAGS.dataset_dir
      ))
      dataset_list.append(dataset_factory.get_dataset(
        FLAGS.dataset_name, 'dev', FLAGS.dataset_dir
      ))
    elif FLAGS.dataset_split_name == "train":
      dataset_list.append(dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir
      ))
    else:
      raise ValueError("Only for train or train_eval")

    ######################
    # Select the network #
    ######################
    network_fn_list = []  # 0 - train network fn; 1 - dev network fn
    if FLAGS.dataset_split_name == "train_eval":
      # Training network fn
      network_fn_list.append(nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset_list[MODE_MAP['train']].num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True))
      # Eval network fn (w/o L2-regularization)
      network_fn_list.append(nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset_list[MODE_MAP['eval']].num_classes - FLAGS.labels_offset),
        is_training=False))
    elif FLAGS.dataset_split_name == "train":
      network_fn_list.append(nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset_list[MODE_MAP['train']].num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True))
    else:
      raise ValueError("Only for train or train_eval")

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True,
        use_grayscale=FLAGS.use_grayscale)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    batch_queue_list = []  # 0 - train batch; 1 - dev batch
    with tf.device(deploy_config.inputs_device()):
      # provider对象根据dataset信息读取数据
      for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        if i == MODE_MAP['train']:
          provider = slim.dataset_data_provider.DatasetDataProvider(
              dataset,
              num_readers=FLAGS.num_readers,
              common_queue_capacity=20 * FLAGS.batch_size,
              common_queue_min=10 * FLAGS.batch_size)
        elif i == MODE_MAP['eval']:
          provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=2*FLAGS.eval_batch_size,
            common_queue_min=FLAGS.eval_batch_size)
        # 获取数据，获取到的数据是单个数据，还需要对数据进行预处理，组合数据
        # provider.get(): Returns a list of tensors specified by the given list of items
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        if i == MODE_MAP['train']:
          # Training set
          train_image_size = FLAGS.train_image_size or network_fn_list[MODE_MAP['train']].default_image_size
          # Pre-processing a single image
          image = image_preprocessing_fn(image, train_image_size, train_image_size)
          print("Training dataset: image shape: {}".format(image.shape))
          # batch images
          images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,    # training set batch size
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)
          labels = slim.one_hot_encoding(
            labels, dataset.num_classes - FLAGS.labels_offset)
          # assemble the batch
          batch_queue = slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * deploy_config.num_clones)
        elif i == MODE_MAP['eval']:
          # Eval Set
          eval_image_size = FLAGS.eval_image_size or network_fn_list[MODE_MAP['eval']].default_image_size
          image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
          print("Eval dataset: image shape: {}".format(image.shape))
          images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.eval_batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=2*FLAGS.eval_batch_size)
          labels = slim.one_hot_encoding(
            labels, dataset.num_classes - FLAGS.labels_offset)  # 算 validation loss 用到
          batch_queue = slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * deploy_config.num_clones)
        else:
          raise ValueError("The length of dataset list is illegal.")
        batch_queue_list.append(batch_queue)
    
    assert(len(batch_queue_list) == len(dataset_list) 
            and len(batch_queue_list) == len(network_fn_list))

    input("Press any key to continue...")

    ############################
    # Select the loss function #
    ############################
    loss_fn = loss_factory.get_loss_fn(FLAGS.loss_fn)

    ####################
    # Define the model #
    ####################
    def clone_fn(network_fn, batch_queue, scope_name):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        loss_fn(
            end_points['AuxLogits'], labels,
            label_smoothing=FLAGS.label_smoothing, weights=0.4,
            scope='{}/aux_loss'.format(scope_name))
      loss_fn(
          logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0,
          scope=scope_name)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    
    # 在这函数中 clone_fn 会被调用 FLAGS.num_clones 次，默认1次
    if FLAGS.dataset_split_name == "train":
      clones = model_deploy.create_clones(deploy_config, 
                                          clone_fn, 
                                          [network_fn_list[MODE_MAP['train']],
                                            batch_queue_list[MODE_MAP['train']],
                                            'train']
                                          )
    elif FLAGS.dataset_split_name == "train_eval":
      # clone_fn definded in create_train_eval_clones() 
      clones =model_deploy.create_train_eval_clones(deploy_config,
                                                    network_fn_list,
                                                    batch_queue_list,
                                                    loss_fn=loss_fn)
    else:
      raise ValueError("Only for train or train_eval")

    input("Continue...")

    #  clone scope format: "training/clone_i" or "eval/clone_(n-1)"                                         
    first_clone_scope = "training/{}".format(deploy_config.clone_scope(0))
    if FLAGS.dataset_split_name == "train_eval":
      first_clone_scope = "train_eval/{}".format(deploy_config.clone_scope(0))
    
    print(first_clone_scope)
    
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points (dict).
    # end_points = [train_end_points, eval_end_points] in 'train_eval' mode
    end_points = clones[0].outputs
    train_end_points = {}
    eval_end_points = {}
    if FLAGS.dataset_split_name == "train_eval" and isinstance(end_points, list):
      train_end_points = end_points[0]
      eval_end_points = end_points[1] 
    else:
      train_end_points = end_points

    print(train_end_points.keys())
    print("===================================")
    print(eval_end_points.keys())

    for end_point in train_end_points:
      x = train_end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))  # Tensor中0元素在所有元素中所占的比例

    # Add summaries for losses.
    print(">> Total losses in {}: {}".format(first_clone_scope, 
                                              len(tf.get_collection(
                                                tf.GraphKeys.LOSSES, first_clone_scope)
                                                )))

    if FLAGS.dataset_split_name == "train_eval":
      for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
        summaries.add(tf.summary.scalar('train_eval/losses/%s' % loss.op.name, loss))
    elif FLAGS.dataset_split_name == "train":
      for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
        summaries.add(tf.summary.scalar('training/losses/%s' % loss.op.name, loss))
    else:
      raise ValueError("Only for train or train_eval")

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    if FLAGS.quantize_delay >= 0:
      contrib_quantize.create_training_graph(quant_delay=FLAGS.quantize_delay)

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      print(">> Num of Training dataset samples: {}".format(dataset_list[MODE_MAP['train']].num_samples))
      input("Continues...")
      learning_rate = _configure_learning_rate(dataset_list[MODE_MAP['train']].num_samples, 
                                                global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    if FLAGS.dataset_split_name == "train_eval":  
      total_loss, clones_gradients, eval_total_loss = model_deploy.optimize_train_eval_clones(
          clones,
          optimizer,
          var_list=variables_to_train)
    elif FLAGS.dataset_split_name == "train":
      total_loss, clones_gradients = model_deploy.optimize_clones(
          clones,
          optimizer,
          var_list=variables_to_train)
    else:
      raise ValueError("Only for train or train_eval")

    input("Continue...")

    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))
    if FLAGS.dataset_split_name == "train_eval":
      summaries.add(tf.summary.scalar('eval_total_loss', eval_total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients, 
                                              global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    ###########################
    # Kicks off the training. #
    ###########################
    # TODO: Evaluation during training
    # Refer to: 
    # https://stackoverflow.com/questions/48898117/how-do-you-run-a-validation-loop-during-slim-learning-train
    # Override the train_step_fn function defined in slim.learning.train()
    def train_step_fn(sess, train_op, global_step, train_step_kwargs):
      if hasattr(train_step_fn, 'step'):
        train_step_fn.step += 1
        #print(">> Train_step_fn.step: {}".format(train_step_fn.step))
      else:
        train_step_fn.step = global_step.eval(sess)
      
      # Calc Training loss
      total_loss, should_stop = slim.learning.train_step(sess, 
                                                          train_op, 
                                                          global_step, 
                                                          train_step_kwargs)

      # # Eval on interval
      if train_step_fn.step and (train_step_fn.step % FLAGS.validation_interval_steps == 0):
        print(">> Start to validation...")
        val_loss_out = sess.run([eval_total_loss])
        print(">> Global step: {}; Validation losses: {}".format(
          train_step_fn.step,
          val_loss_out
        ))
      
      return [total_loss, should_stop]

    if FLAGS.dataset_split_name == "train_eval":
      slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        train_step_fn=train_step_fn,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None)
    else:
      slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None)


if __name__ == '__main__':
  tf.app.run()
