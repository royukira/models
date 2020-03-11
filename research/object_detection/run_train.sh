tf_version="tf14"
train_path="/roy_work/Object_detection_API/training/"
pipeline_path="$train_path/ssdlite_mobilenet_v3_large_320x320_coco.config"
num_train_steps=200000
num_eval_steps=100

#conda activate tf14
python model_main.py --model_dir=$train_path --pipeline_config_path=$pipeline_path --alsologtostderr --num_train_steps=$num_train_steps --num_eval_steps=$num_eval_steps
