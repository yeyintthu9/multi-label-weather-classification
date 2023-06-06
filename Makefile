.PHONY : install-prerequisite-mlflow install-prerequisite-docker build-docker-env setup-docker-env setup-mlflow-env run-mlflow

OPERATION?=train
CONFIG?=$(shell pwd)/config/config.yml

export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu116

download-data:
	./bash/download-data.sh
	
install-prerequisite-mlflow:
	./bash/install-prerequisite-mlflow-env.sh

install-prerequisite-docker:
	sudo ./bash/install-prerequisite-docker-env.sh

build-docker-env: 
	sudo docker build -t multi-label-classification -f docker/Dockerfile .

setup-docker-env: build-docker-env
	sudo docker run -it --gpus all  -v $(shell pwd)/:/home/multi-label-weather-classification/ \
		multi-label-classification /bin/bash || true	

setup-mlflow-env:
	mlflow run . -e check-status

# an extra command is needed for explicitly setting experiment name and run name due to an issue of mlflow
# Issue: https://github.com/mlflow/mlflow/issues/799
run-as-mlflow-project: setup-mlflow-env
	mlflow run . -P operation_type=$(OPERATION) -P config_path=$(CONFIG) \
		--experiment-name $(shell yq eval '.mlflow_exp_name' $(CONFIG)) \
		--run-name $(shell date +"%H:%M:%S_%d-%m-%Y")