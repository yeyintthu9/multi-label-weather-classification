# -*- coding: utf-8 -*-
"""
@Author: Ye Yint Thu
@Email: yeyintthu536@gmail.com
"""

from copy import deepcopy
import cv2
import numpy as np
import os
from sklearn.metrics import classification_report
import torch
import torchvision
from typing import List, Iterator, Tuple, Dict, Union
from random import uniform
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging


def criterion(loss_fn, preds, gts, device):
    losses = torch.zeros((len(preds.keys())), device=device)
    for i, key in enumerate(preds):
        losses[i] = loss_fn(preds[key], torch.unsqueeze(gts[key], 1).float().to(device))
    return torch.mean(losses)


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    lr_rate: float,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    milestones: List[int],
    saved_dir: str,
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, Iterator]:
    """Train the model with multi-steps learning rates for a given order of epochs
        and return the best checkpointed model, training and validation losses

    Args:
        model (torch.nn.Module): Nn module that represents a torch model
        device (torch.device): Device on which training process will be run
        lr_rate (float): Initial learning rate for training
        num_epochs (int): Number of epochs
        train_loader (torch.utils.data.DataLoader): Data loader for training set
        val_loader (torch.utils.data.DataLoader): Data loader for testing set
        Milestones (List[int]): Milestones of epochs for multi-steps learning process
        saved_dir (str): Directory to save model checkpoints
        logger (logging.Logger): Logging object for writing out logs
    Returns:
        Tuple[torch.nn.Module, Iterator]: Best model and, training and validation losses
            through epochs
    """
    # define scheduled optimizer and loss func
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    loss_func = torch.nn.BCELoss()
    # initialize losses and model
    train_epoch_loss = []
    validation_epoch_loss = []
    best_loss = 999.00
    model = model.to(device)
    best_model = deepcopy(model)
    # train and validate per epoch
    for epoch in range(num_epochs):
        for data_loader in [
            {"type": "train", "loader": train_loader},
            {"type": "val", "loader": val_loader},
        ]:
            losses = []
            if data_loader["type"] == "train":
                model.train()
            else:
                model.eval()
            n_total_steps = len(data_loader["loader"])  # type: ignore
            for i, samples in enumerate(data_loader["loader"]):  # type: ignore
                images = samples["image_tensors"].to(device)
                gts = samples["labels"]
                scheduler.optimizer.zero_grad()  # type: ignore
                with torch.set_grad_enabled(data_loader["type"] == "train"):
                    preds = model(images)
                    loss = criterion(loss_func, preds, gts, device=device)
                    losses.append(loss.item())
                    if data_loader["type"] == "train":
                        loss.backward()
                        scheduler.optimizer.step()  # type: ignore
                current_loss = torch.tensor(losses).mean().item()
                logger.info(
                    (
                        f'Epoch[{epoch+1}/{num_epochs}], Phase-> {data_loader["type"]},'
                        + f" Step[{i+1}/{n_total_steps}], Loss: {current_loss:.4f}"
                    )
                )
                if (i + 1) % (int(n_total_steps / 1)) == 0:
                    if data_loader["type"] == "val":
                        validation_epoch_loss.append(current_loss)
                        logger.info(
                            f"Current loss :{current_loss} \nBest loss :{best_loss}"
                        )
                        if current_loss < best_loss:
                            best_loss = current_loss
                            best_model = deepcopy(model)
                            logger.info("Updating checkpoint..")
                            os.makedirs(saved_dir, exist_ok=True)
                            torch.save(
                                best_model,
                                os.path.join(saved_dir, "best_model.pt"),
                            )
                        if epoch == num_epochs - 1:
                            torch.save(
                                model,
                                os.path.join(saved_dir, "last_model.pt"),
                            )
                    if data_loader["type"] == "train":
                        train_epoch_loss.append(current_loss)
    return best_model, zip(*[train_epoch_loss, validation_epoch_loss])


def evaluate_model(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    classes: List,
) -> Dict[str, Dict[str, Union[Dict, float]]]:
    """Evaluate multi-label binary classifier and return classification reports
        for classifiers

    Args:
        model (torch.nn.Module): Nn module representation of multi-label
            binary classification model
        device (torch.device): Device on which evalution will be run
        data_loader (_type_): Dataloader for evalution
        classes (List): List of classes

    Returns:
        Dict[str, Dict[str, Union[Dict, float]]]: Dict of classification report dicts
            for classifiers' metrics
    """
    model.eval()
    classes_gts, classes_preds = {}, {}  # type: ignore
    for class_ in classes:
        classes_gts[class_] = []
        classes_preds[class_] = []
    for _, samples in tqdm(enumerate(data_loader), desc="Evaluating.."):
        images = samples["image_tensors"].to(device)
        gts = samples["labels"]
        for class_, val in gts.items():
            classes_gts[class_] = classes_gts[class_] + [
                int(label) for label in val.tolist()
            ]
        with torch.no_grad():
            preds = model(images)
        for class_, val in preds.items():
            classes_preds[class_] = classes_preds[class_] + [
                round(prob[0]) for prob in val.cpu().numpy()
            ]

    classification_reports_dict = {
        class_: classification_report(
            classes_gts[class_],
            classes_preds[class_],
            target_names=[f"Not {class_}", class_],
            output_dict=True,
        )
        for class_ in classes
    }
    return classification_reports_dict


def predict(
    model: torch.nn.Module,
    img_paths: List[str],
    transforms: torch.nn.Module,
    device: torch.device,
    classes: List[str],
) -> Dict[str, List]:
    """Predict values from multi-labels binary classification model
    for a given set of image paths

    Args:
        model (torch.nn.Module): Nn module representation of multi-label
            binary classification model
        img_paths (List[str]): Paths of images
        transforms (torch.nn.Module): Transformation pipeline before feeding
            into the model
        device (torch.device): Device on which predction will be run
        classes (List[str]): Classes of the dataset

    Returns:
        Dict[str, List]: Dict of predcitions for each of binary classifier
    """
    predicted_classes: Dict[str, List] = {class_: [] for class_ in classes}

    for img_path in tqdm(img_paths):
        img = Image.open(img_path).convert("RGB")
        img = transforms(img)
        img = torch.stack([img]).to(device)
        with torch.no_grad():
            preds = model(img)
        for class_, pred in preds.items():
            predicted_classes[class_].append(round(pred.tolist()[0][0]))
    return predicted_classes


def validate_model(
    model: torch.nn.Module, config: Dict, classes: List, logger: logging.Logger
) -> None:
    """Validate and visualize predicted results and compare with ground truth labels

    Args:
        model (torch.nn.Module): Nn module representation of multi-label
            binary classification model
        config (Dict): Config params
        classes (List): Classes of the dataset
        logger (logging.Logger): Logger to write out validation logs
    """
    # create directory to save validated images
    os.makedirs(config["validated_images_dir"], exist_ok=True)
    # retrieve image paths, ground truth labels and classes
    img_paths: List = []
    with open(config["dataset"]["val_path"], "r") as f:
        val_annos = [sample.replace("\n", "") for sample in f.readlines()]
    with open(config["dataset"]["classes_path"], "r") as f:
        classes = f.read().rstrip().split("\n")
    img_paths = []
    gts: Dict[str, List] = {class_: [] for class_ in classes}
    for val_sample in val_annos:
        file_path, labels = val_sample.split()[0], [
            int(label) for label in val_sample.split()[1:]
        ]
        img_paths.append(os.path.join(config["dataset"]["val_images_root"], file_path))
        for i, class_ in enumerate(classes):
            gts[class_].append(labels[i])
    # create transformation pipelines
    input_size = config["backbone"]["input_size"]
    mean = config["dataset"]["mean"]
    std = config["dataset"]["std"]
    transforms = get_transforms(input_size, mean, std, train_pipeline=False)
    # load model in eval mode
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = torch.load(config["evaluation"]["model_file"],map_location=device)
    model.eval()
    model.to(device)
    img_paths = img_paths
    logger.info(f"Number of images for validation: {len(img_paths)}")
    # predict and visualize for validation
    logger.info("Inferencing..")
    preds = predict(model, img_paths, transforms, device, classes)
    logger.info(
        f'Visualizing and saving resut images in {config["validated_images_dir"]}..'
    )
    visualize_preds_vs_gts(
        img_paths, preds, gts, classes, saved_images_dir=config["validated_images_dir"]
    )
    logger.info("Validation process done!")


def visualize_preds_vs_gts(
    img_paths: List[str],
    preds: Dict[str, List],
    gts: Dict[str, List],
    classes: List[str],
    saved_images_dir: str,
) -> None:
    """Visualize the comparison of prediction and ground truth values on images

    Args:
        img_paths (List[str]): Paths of images for validation
        preds (Dict[str, List]): Prediction results of binary classifiers
            for validated images
        gts (Dict[str, List]): Ground truth values of binary classifiers
            for validated images
        classes (List[str]): Classes of dataset
        saved_images_dir (str): Director to save validated images
    """
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img_2 = np.ones((600, 600, 3), dtype=np.uint8) * 255
        txt_x = 90
        cv2.putText(
            img_2,
            "Gt",
            (20, 40),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            img_2,
            "Pred",
            (20, 60),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 0, 0),
            1,
        )
        for j, class_ in enumerate(classes):
            txt_y = 20
            label_color = (
                (0, 255, 0) if gts[class_][i] == preds[class_][i] else (0, 0, 255)
            )
            cv2.putText(
                img_2,
                class_,
                (txt_x, txt_y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 0, 0),
                1,
            )
            txt_y += 20
            cv2.putText(
                img_2,
                "yes" if gts[class_][i] else "no",
                (txt_x, txt_y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 255, 0),
                1,
            )
            txt_y += 20
            cv2.putText(
                img_2,
                "yes" if preds[class_][i] else "no",
                (txt_x, txt_y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                label_color,
                1,
            )
            txt_x += 90

        x_offset = y_offset = 150
        img_2[
            y_offset : y_offset + img.shape[0],  # noqa
            x_offset : x_offset + img.shape[1],  # noqa
        ] = img
        cv2.imwrite(
            os.path.join(saved_images_dir, f"validated_{Path(img_path).name}"), img_2
        )


def get_transforms(
    input_size: Tuple, mean: List, std: List, train_pipeline: bool = True
) -> torch.nn.Module:
    """Construct tarnsformation pipeline for train and validation/test operations

    Args:
        input_size (Tuple): Input size of model
        mean (List): Mean of the dataset
        std (List): Std of the dataset
        train_pipeline (bool, optional): Flag for training process. Defaults to True.

    Returns:
        torch.nn.Module: A stack of transformaton operations based on
            pipeline type (train/val)
    """
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )
    if train_pipeline:
        augmented_transform = torchvision.transforms.RandomChoice(
            [
                torchvision.transforms.RandomApply(
                    torch.nn.ModuleList([torchvision.transforms.RandomRotation(30)]),
                    p=0.4,
                ),
                torchvision.transforms.RandomApply(
                    torch.nn.ModuleList(
                        [
                            torchvision.transforms.CenterCrop(
                                size=int(round(uniform(0.1, 0.3), 2) * input_size[0])
                            )
                        ]
                    ),
                    p=0.4,
                ),
                torchvision.transforms.RandomHorizontalFlip(p=0.4),
            ]
        )
        transform_pipeline = torchvision.transforms.Compose(
            [augmented_transform, transforms]
        )

    else:
        transform_pipeline = transforms
    return transform_pipeline
