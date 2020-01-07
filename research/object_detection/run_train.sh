tf_version="tf14"
train_path="/roy_work/Object_detection_API/training/"
pipeline_path="$train_path/ssd_mobilenet_v2_coco_head.config"
num_train_steps=2000000
num_eval_steps=1000

#conda activate tf14
python model_main.py --model_dir=$train_path --pipeline_config_path=$pipeline_path --alsologtostderr --num_train_steps=$num_train_steps --num_eval_steps=$num_eval_steps
