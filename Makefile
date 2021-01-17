IMAGES_NAME="bert_experiment"
VOLUME_DIR="/repository"
LOGS_DIR = "logs"

GPU_DEVICE="all"
#GPU_DEVICE="device=0"
#GPU_DEVICE="device=1"

DATA_DIR="./datasets/sentiment"
OUTPUT_DIR="checkpoints/sentiment/2"

MODEL_NAME="bert-base-chinese"
#MODEL_NAME="ckiplab/bert-base-chinese"
#MODEL_NAME="ckiplab/albert-base-chinese"
#MODEL_NAME="ckiplab/albert-tiny-chinese"

CONTAINER_RAM="4g" # Units : k/m/g
CONTAINER_CPU="4"

# 指令
build_images:
	@docker build --build-arg VOLUME_DIR=$(VOLUME_DIR) -t $(IMAGES_NAME) .

run_fine_tuning_sentiment:
	@docker run --gpus $(GPU_DEVICE) --rm \
		--log-opt max-size=256m --log-opt max-file=30 \
		--memory=$(CONTAINER_RAM) --cpus=$(CONTAINER_CPU) \
		-v $(CURDIR):$(VOLUME_DIR) $(IMAGES_NAME) \
		python core/run_sentiment.py \
			--task_name=sentiment \
			--do_train=True \
			--do_eval=False \
			--do_demo=False \
			--data_dir=$(DATA_DIR) \
			--num_epochs=2 \
			--model_name=$(MODEL_NAME) \
			--output_dir=$(OUTPUT_DIR)

run_eval_sentiment:
	@docker run --gpus $(GPU_DEVICE) --rm \
		--log-opt max-size=256m --log-opt max-file=30 \
		--memory=$(CONTAINER_RAM) --cpus=$(CONTAINER_CPU) \
		-v $(CURDIR):$(VOLUME_DIR) $(IMAGES_NAME) \
		python core/run_sentiment.py \
			--task_name=sentiment \
			--do_train=False \
			--do_eval=True \
			--do_demo=False \
			--data_dir=$(DATA_DIR) \
			--output_dir=$(OUTPUT_DIR)

run_demo_sentiment:
	@docker run --gpus $(GPU_DEVICE) --rm -it \
		--log-opt max-size=256m --log-opt max-file=30 \
		--memory=$(CONTAINER_RAM) --cpus=$(CONTAINER_CPU) \
		-v $(CURDIR):$(VOLUME_DIR) $(IMAGES_NAME) \
		python core/run_sentiment.py \
			--task_name=sentiment \
			--do_train=False \
			--do_eval=False \
			--do_demo=True \
			--output_dir=$(OUTPUT_DIR)

show_tensorBoard:
	@eval tensorboard --host 0.0.0.0 --logdir $(LOGS_DIR)

delete_log_dir:
	@eval sudo rm -r $(LOGS_DIR)