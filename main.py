# -*- coding: utf-8 -*-
"""
@Author: Ye Yint Thu
@Email: yeyintthu536@gmail.com
"""

import argparse
from enum import Enum
import mlflow
import os
import pickle
import pandas as pd
import yaml
import torch
from datetime import datetime
from typing import Iterator, List, Dict, Union
import logging

# import custom packages
from dataset.multi_label_weather_dataset import MultiLabelWeatherDataset
from model.model import MultiLabelBinaryClassifier
from utils.utils import train_model, evaluate_model, validate_model, get_transforms


# define dataset types
class OperationTypes(Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    VALIDATE = "validate"
    TRAIN_EVAL = "train_eval"


class MLFlowExperiment:
    """MLFLowExperiment contains functions to log output of training
    as mlflow metrics and artifacts

    Args:
        config (Dict): Config params for logging
        log_dir (str): Directory to save some mlflow artifacts of the experiment
    """

    def __init__(
        self, config: Dict, log_dir: str = "mlflow_logs"
    ) -> None:
        self.exp_name = config["mlflow_exp_name"]
        self.config = config
        self.log_dir = log_dir
        experiment = mlflow.get_experiment_by_name(self.exp_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
        else:
            self.experiment_id = mlflow.create_experiment(self.exp_name)

    def init_run(self):
        """Start new MLFlow run"""
        mlflow.start_run(
            run_name=datetime.now().strftime("%H:%M:%S_%d-%M-%Y"),
            experiment_id=self.experiment_id
        )

    def terminate_run(self):
        """End active MLFlow run"""
        mlflow.end_run()

    def log_params(self):
        """Log a batch of training params for the current run"""
        mlflow.log_params(
            {
                "Batch size": config["dataset"]["batch_size"],
                "Backbone type": config["backbone"]["type"],
                "Frozen backbone layers": config["backbone"]["frozen_layers"],
                "Input size": config["backbone"]["input_size"],
                "Learning rate": config["training"]["lr"],
                "Number of epochs": config["training"]["epochs"],
                "Milestones": config["training"]["milestones"],
            }
        )

    def log_accuracy(self, classifiers_accuracies_dict: Dict[str, float]):
        """Log accuracies of classifiers for the current run"""
        mlflow.log_metrics(classifiers_accuracies_dict)

    def log_config(self, config_path: str = "config.yml"):
        """Log config file as MLFLow artifact

        Args:
            config_path (str, optional): Path of config file which is used in training.
                Defaults to "config.yml".
        """
        os.makedirs(self.log_dir, exist_ok=True)
        source_config_path = os.path.join(self.log_dir, config_path)
        with open(source_config_path, "w") as con_f:
            yaml.dump(config, con_f)
        mlflow.log_artifact(source_config_path, artifact_path=config_path)

    def log_losses(self, losses: Iterator):
        """Log BCE losses of current run

        Args:
            losses (Iterator): _description_
        """
        for i, loss_info in enumerate(losses):
            mlflow.log_metrics(
                {"BCE train loss": loss_info[0], "BCE val Loss": loss_info[1]},
                i,
            )

    def log_model(self, model: torch.nn.Module):
        """Log model of current run

        Args:
            model (torch.nn.Module): Pytorch model
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path="best-model",
            pickle_module=pickle,
        pip_requirements=["torch==1.13.1"],
        )


def log_gitstyle_classification_report(
    classifier_name: str, report_dict: Dict, logger: logging.Logger
) -> None:
    """Format classification report as github style table and print it out
    along with  overall accuracy

    Args:
        classifier_name (str): Name of classifier from which classification
            report is generated
        report_dict (Dict): Dictionary of metrics of a classification report
        logger (logging.Logger): Logger to write out report logs
    """
    accuracy = report_dict.pop("accuracy")
    report_df = pd.DataFrame(report_dict).transpose()
    logger.info(
        f"\n\nClassification report for {classifier_name} weather classification model:"
        + f"\n\n{report_df.to_markdown(index=True,tablefmt='github')}\n\n"
        + f"  Overall accuracy : {accuracy}\n\n"
    )


def get_dataloader(
    config: Dict, classes: List, transforms, for_train: bool = True
) -> torch.utils.data.DataLoader:
    """Get dataloader for the given process train/val

    Args:
        config (Dict): Config params
        classes (List): List of classes
        transforms (torch.nn.Module): Transformation pipeline
        for_train (bool, optional): Flag for training process. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Dataloader for the given process type
    """
    is_shuffle = True
    if for_train:
        dataset = MultiLabelWeatherDataset(
            config["dataset"]["train_path"],
            config["dataset"]["train_images_root"],
            transforms,
            classes,
        )
    else:
        dataset = MultiLabelWeatherDataset(
            config["dataset"]["val_path"],
            config["dataset"]["val_images_root"],
            transforms,
            classes,
        )
        is_shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=is_shuffle,
        num_workers=config["num_workers"],
    )


def train(
    model: torch.nn.Module,
    config: Dict,
    device: torch.device,
    mlflow_tracker: MLFlowExperiment,
    eval: bool,
    logger: logging.Logger,
    output_dir: str = "outputs/default_exp/",
):
    """Train the model with provided configuration params

    Args:
        model (torch.nn.Module): Nn module representation of multi-label
            binary classifier
        config (Dict): Dict of config params
        device (torch.device): Device on which training will be run
        mlflow_tracker (MLFlowExperiment): Mlflow tracker for logging metrics,
            params and artifacts
        eval (bool): Operate evalution or not
        logger(logging.Logger): Logging object for writing out logs
        output_dir (str, optional): directory to save outputs from trianing process.
            Default to `outputs/default_exp/`
    """
    # create transformation pipelines
    input_size = config["backbone"]["input_size"]
    mean = config["dataset"]["mean"]
    std = config["dataset"]["std"]

    train_transforms = get_transforms(input_size, mean, std)
    val_transforms = get_transforms(input_size, mean, std, train_pipeline=False)
    # create datasets and dataloaders for training and validation sets
    train_loader = get_dataloader(config, classes, train_transforms)
    val_loader = get_dataloader(config, classes, val_transforms, for_train=False)

    saved_model_dir = os.path.join(output_dir, "saved_model")

    mlflow_tracker.init_run()
    mlflow_tracker.log_params()
    # training
    best_model, losses = train_model(
        model,
        device,
        config["training"]["lr"],
        config["training"]["epochs"],
        train_loader,
        val_loader,
        config["training"]["milestones"],
        saved_model_dir,
        logger,
    )
    

    # write training and validation losses to tensorboard
    mlflow_tracker.log_losses(losses)
    mlflow_tracker.log_config()
    mlflow_tracker.log_model(model)

    if eval:
        accuracies_dict = {}
        eval_reports_dict = evaluate(best_model, config, device, classes)
        for classifier_name, eval_report in eval_reports_dict.items():  # type: ignore
            accuracies_dict[classifier_name] = eval_report.get("accuracy")
            log_gitstyle_classification_report(
                classifier_name, eval_report, operations_logger
            )
        mlflow_tracker.log_accuracy(accuracies_dict)  # type: ignore
    mlflow_tracker.terminate_run()


def evaluate(
    model: torch.nn.Module,
    config: Dict,
    device: torch.device,
    classes: List,
) -> Dict[str, Dict[str, Union[float, Dict]]]:
    """Evaluate model and generate classification reports for the validation set

    Args:
        model (torch.nn.Module): Nn module representation of multi-label
            binary classifier
        config (Dict): Dict of config params
        device (torch.device): Device on which evaluation will be run
        classes (List): List of classes of the dataset

    Returns:
        Dict[str, Dict[str, Union[float, Dict]]]: Dictionary of dictionaries
            for classification reports of classifiers

    """
    # create transformation pipelines
    input_size = config["backbone"]["input_size"]
    mean = config["dataset"]["mean"]
    std = config["dataset"]["std"]
    val_transforms = get_transforms(input_size, mean, std, train_pipeline=False)
    # create datasets and dataloaders for training and validation sets
    val_loader = get_dataloader(config, classes, val_transforms, for_train=False)
    classification_reports = evaluate_model(model, device, val_loader, classes)
    return classification_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Training and/or validation of multi-label binary classifier"
            + "for multi-labelweather prediction"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "operation_type",
        help=(
            "Use one of the follwing operations\n"
            + "  `train`     : run training operation\n"
            + "  `evaluate`  : run evaluation operation(generate metrics)\n"
            + "  `train_eval`: run training operation then evalution operation\n"
            + "  `validate`  : run validation operatoin(visualize Gts vs Preds)"
        ),
    )
    parser.add_argument(
        "--config_path",
        default="./config/config.yml",
        help="Config yml file for the training/evaluation/validation",
    )
    args = parser.parse_args()
    if not os.path.exists(args.config_path):
        raise FileNotFoundError("Config file is not found!")
    if not args.config_path.endswith(".yml"):
        raise NotImplementedError(
            "Unsupported config file type! Please use yml config file!"
        )

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
    exp_ext = f'{datetime.now().strftime("%H:%M:%S_%d-%m-%Y")}_{args.operation_type}'
    output_dir = os.path.join(config["saved_dir"], exp_ext)
    mlflow_log_dir = os.path.join(output_dir, "mlflow-log")
    mlflow_tracker = MLFlowExperiment(config, mlflow_log_dir)
    with open(config["dataset"]["classes_path"], "r") as f:
        classes = f.read().rstrip().split("\n")

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # create classifier
    classifier = MultiLabelBinaryClassifier(
        classes, config["backbone"]["type"], config["backbone"]["frozen_layers"], device
    )

    
    log_path = os.path.join(config["log_dir"], f"{exp_ext}.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.WARNING,
        format="%(asctime)s :: %(message)s",
    )
    # Create logger, set level, and add stream handler
    operations_logger = logging.getLogger()
    operations_logger.setLevel(logging.INFO)
    operations_stream_handler = logging.StreamHandler()
    operations_logger.addHandler(operations_stream_handler)

    
    
    # run operation based on operation type
    if args.operation_type == OperationTypes.TRAIN.value:
        
        train(
            classifier,
            config,
            device,
            mlflow_tracker,
            eval=False,
            logger=operations_logger,
            output_dir=output_dir,
        )
    elif args.operation_type == OperationTypes.EVALUATE.value:
        eval_classifier = torch.load(config["evaluation"]["model_file"],map_location=device)
        evaluation_reports = evaluate(eval_classifier, config, device, classes)
        for classifier_name, eval_report in evaluation_reports.items():  # type: ignore
            log_gitstyle_classification_report(
                classifier_name, eval_report, operations_logger
            )
    elif args.operation_type == OperationTypes.TRAIN_EVAL.value:
        mlflow_tracker = MLFlowExperiment(config, mlflow_log_dir)
        train(
            classifier,
            config,
            device,
            mlflow_tracker,
            eval=True,
            logger=operations_logger,
            output_dir=output_dir,
        )
    elif args.operation_type == OperationTypes.VALIDATE.value:
        validate_model(classifier, config, classes, operations_logger)
