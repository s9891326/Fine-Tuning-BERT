IMAGES_NAME="bert_experiment"
VOLUME_DIR="/repository"
LOGS_DIR = "tensorBoard_logs"
TASK_NAME = "sentiment"

GPU_DEVICE="all"
#GPU_DEVICE="device=0"
#GPU_DEVICE="device=1"

DATA_DIR="datasets/old_sentiment"
OUTPUT_DIR="checkpoints-out/sentiment/2"
DEPLOY_DIR="checkpoints-dep/sentiment/2"

SAVEDMODEL="savedmodel"
#SAVEDMODEL="None"
TENSORRT="tensorRT"
#TENSORRT="None"
#TFLITE="tflite"
#TFLITE="None"

SERVING_DIR="serving/models/sentiment"

#MODEL_TYPE="custom"
MODEL_TYPE="tf-hub"
#MODEL_TYPE="origin"

MODEL_NAME="bert-base-chinese"
#MODEL_NAME="ckiplab/bert-base-chinese"
#MODEL_NAME="ckiplab/albert-base-chinese"
#MODEL_NAME="ckiplab/albert-tiny-chinese"

USE_DEV_DATASET=False # --use_dev_dataset=$(USE_DEV_DATASET)
LOAD_MODEL_DIR=None  # --load_model_dir=$(LOAD_MODEL_DIR)
MIX_NUMBER=0  # --mix_number=$(MIX_NUMBER)

export NODE_NAME=tensorflow_node

# 指令
build_images:
	@docker build --build-arg VOLUME_DIR=$(VOLUME_DIR) -t $(IMAGES_NAME) .

run_fine_tuning_sentiment:
	@docker run --gpus $(GPU_DEVICE) --rm \
		-v $(CURDIR):$(VOLUME_DIR) $(IMAGES_NAME) \
		python core/run_sentiment.py \
			--task_name=$(TASK_NAME) \
			--do_train=True \
			--save_format=$(SAVEDMODEL) \
			--save_format=$(TENSORRT) \
			--model_type=$(MODEL_TYPE) \
			--data_dir=$(DATA_DIR) \
			--model_name=$(MODEL_NAME) \
			--output_dir=$(OUTPUT_DIR) \
			--deploy_dir=$(DEPLOY_DIR)

run_load_model_to_train:
	@docker run --gpus $(GPU_DEVICE) --rm \
		-v $(CURDIR):$(VOLUME_DIR) $(IMAGES_NAME) \
		python core/run_sentiment.py \
			--task_name=$(TASK_NAME) \
			--do_train=True \
			--save_format=$(SAVEDMODEL) \
			--save_format=$(TENSORRT) \
			--data_dir=$(DATA_DIR) \
			--load_model_dir=$(LOAD_MODEL_DIR) \
			--mix_number=$(MIX_NUMBER) \
			--model_name=$(MODEL_NAME) \
			--output_dir=$(OUTPUT_DIR) \
			--deploy_dir=$(DEPLOY_DIR)

run_test_sentiment:
	@docker run --gpus $(GPU_DEVICE) --rm \
		-v $(CURDIR):$(VOLUME_DIR) $(IMAGES_NAME) \
		python core/run_sentiment.py \
			--task_name=$(TASK_NAME) \
			--do_test=True \
			--use_dev_dataset=$(USE_DEV_DATASET) \
			--model_type=$(MODEL_TYPE) \
			--data_dir=$(DATA_DIR) \
			--output_dir=$(OUTPUT_DIR)

run_inference_sentiment:
	@docker run --gpus $(GPU_DEVICE) --rm -it \
		-v $(CURDIR):$(VOLUME_DIR) $(IMAGES_NAME) \
		python core/run_sentiment.py \
			--task_name=$(TASK_NAME) \
			--do_inference=True \
			--model_type=$(MODEL_TYPE) \
			--output_dir=$(OUTPUT_DIR)

show_tensorBoard:
	@eval tensorboard --host 0.0.0.0 --logdir $(LOGS_DIR)

delete_log_dir:
	@eval sudo rm -r $(LOGS_DIR)

create_load_dataset:
	@python utils/create_load_dataset.py --mix_number $(MIX_NUMBER)

deploy_model:
	@eval bash bin/run_serving.sh

remove_deploy_model:
	@docker stop $(NODE_NAME)
	@docker rm $(NODE_NAME)
