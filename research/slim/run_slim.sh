
# Env
CONDA_PATH="/anaconda3/envs"
TF_VERSION="tf14"
PYTHON="${CONDA_PATH}/${TF_VERSION}/bin/python"


# Basic Params
train_dir="/roy_work/Classification/training"                # Dir where checkpoints and event logs are written to
pretrain_dir="/roy_work/Classification/pretrained_model"     # Dir of pretrained models
model_name="mobilenet_v2"                                    # model's name, like mobilenet_v2, inception_v3, etc.
model_version="mobilenet_v2_1.0_224"                         # model's version, like mobilenet_v2_1.4_224
checkpoint_path="${pretrain_dir}/${model_version}/${model_version}.ckpt"  # TODO: set as your model.cpkt
dataset_name="pci_HeadHat_dav4_cls"                          # TODO: set as your custom dataset
dataset_dir="/roy_work/Classification/data/${dataset_name}"  # Dir of dataset
mode="train"
batch_size=32                                                # The number of samples in each batch. Default 32
max_number_of_steps=200000                                   # The maximum number of training steps.
use_grayscale="False"                                        # Whether to convert input images to grayscale.
labels_offset=0                                              # An offset for the labels in the dataset. This flag is primarily used to evaluate the VGG and ResNet architectures which do not use a background class for the ImageNet dataset.
train_image_size=300                                         # The shape of a train image; Default 224


# Optimization Params
weight_decay=0.00004  # The weight decay on the model weights. Default 0.00004
optimizer="adam"     # The name of the optimizer, one of "adadelta", "adagrad", "adam","ftrl", "momentum", "sgd" or "rmsprop". Default 'rmsprop'
adadelta_rho=0.95  # The decay rate for adadelta. Default 0.95
adagrad_initial_accumulator_value=0.1  # Starting value for the AdaGrad accumulators. Default 0.1
adam_beta1=0.9  # The exponential decay rate for the 1st moment estimates. Default 0.9
adam_beta2=0.999  # The exponential decay rate for the 2nd moment estimates.Default 0.999
opt_epsilon=1.0  # Epsilon term for the optimizer.Default 1.0
ftrl_learning_rate_power=-0.5  # The learning rate power. Default -0.5
ftrl_initial_accumulator_value=0.1  # Starting value for the FTRL accumulators. 0.1
ftrl_l1=0.0  # The FTRL l1 regularization strength. 0.0
ftrl_l2=0.0  # The FTRL l2 regularization strength. 0.0
momentum=0.9  # The momentum for the MomentumOptimizer and RMSPropOptimizer. 0.9
rmsprop_momentum=0.9  # 'Momentum. 0.9
rmsprop_decay=0.9  # 'Decay term for RMSProp. 0.9
quantize_delay=-1  # Number of steps to start quantized training. Set to -1 would disable quantized training. -1


# Learning Rate Params
learning_rate_decay_type="cosine_decay_with_warmup" # Specifies how the learning rate is decayed. One of "fixed", "exponential","polynomial" or "cosine_decay_with_warmup" . Default 'exponential'
learning_rate=0.005  # Initial learning rate. Default 0.01 
end_learning_rate=0.0001  # The minimal end learning rate used by a polynomial decay learning rate.
label_smoothing=0.0  # The amount of label smoothing.
learning_rate_decay_factor=0.94  # Learning rate decay factor.
num_epochs_per_decay=2.0  #  Number of epochs after which learning rate decays. Note: this flag counts epochs per clone but aggregates per sync replicas. So 1.0 means that each clone will go over full epoch individually, but replicas will go once across all replicas.
sync_replicas="False"  # Whether or not to synchronize the replicas during training.
replicas_to_aggregate=1  # The Number of gradients to collect before updating params.
moving_average_decay="None"  # The decay to use for the moving average. If left as None, then moving averages are not used.
warmup_learning_rate=0.001  # Initial learning rate for warm up (Only for cosine_with_warmup)
warmup_steps=1000  # Number of warmup steps. (Only for cosine_with_warmup)


# Fine-tuning Params
num_readers=""  # The number of parallel readers that read data from the dataset. Default 4
num_preprocessing_threads=""  # The number of threads used to create the batches. Default 4
log_every_n_steps=""  # The frequency with which logs are print.  Default 10
save_summaries_secs=""  # The frequency with which summaries are saved, in seconds. Default 600
save_interval_secs=""   # The frequency with which the model is saved, in seconds. Default 600
trainable_scopes=""            # For fine-tuning one only want train a sub-set of layers 
checkpoint_exclude_scopes="MobilenetV2/Logits"   # When we fine-tune a model on a new task with a different number of output labels, we wont be able restore the final logits (classifier) layer, so we need to use this flag to hinder certain variables from being loaded. Consequently, the flags --checkpoint_path and --checkpoint_exclude_scopes are only used during the 0-th global step (model initialization).


# ============================  Run  ===========================================
printf "$PYTHON /roy_work/Object_detection_API/models/research/slim/train_image_classifier.py \
    --train_dir=${train_dir} \
    --dataset_dir=${dataset_dir} \
    --dataset_name=${dataset_name} \
    --dataset_split_name=${mode} \
    --train_image_size=${train_image_size}\
    --model_name=${model_name} \
    --checkpoint_path=${checkpoint_path} \
    --learning_rate_decay_type=${learning_rate_decay_type} \
    --learning_rate=${learning_rate} \
    --warmup_learning_rate=${warmup_learning_rate} \ 
    --warmup_steps=${warmup_steps} \
    --optimizer=${optimizer} \
    --checkpoint_exclude_scopes=${checkpoint_exclude_scopes}   "

$PYTHON /roy_work/Object_detection_API/models/research/slim/train_image_classifier.py \
    --train_dir=${train_dir} \
    --dataset_dir=${dataset_dir} \
    --dataset_name=${dataset_name} \
    --dataset_split_name=${mode} \
    --train_image_size=${train_image_size} \
    --model_name=${model_name} \
    --checkpoint_path=${checkpoint_path} \
    --learning_rate_decay_type=${learning_rate_decay_type} \
    --learning_rate=${learning_rate} \
    --warmup_learning_rate=${warmup_learning_rate} \
    --warmup_steps=${warmup_steps} \
    --optimizer=${optimizer} \
    --checkpoint_exclude_scopes=${checkpoint_exclude_scopes}
