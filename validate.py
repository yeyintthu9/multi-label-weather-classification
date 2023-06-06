import os
from typing import Dict, List
import yaml
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import argparse
from model.model import MultiLabelBinaryClassifier


def predict(config, img_paths, weather_classes):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = MultiLabelBinaryClassifier(
        weather_classes, frozen_layers=config["frozen_layers"], device=device
    )
    model.eval()
    model.load_state_dict(torch.load(config["weights"]))
    model.to(config["device"])
    predicted_weathers = {class_: [] for class_ in weather_classes}
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        img = transforms(img)
        img = torch.stack([img]).to(config["device"])
        with torch.no_grad():
            preds = model(img)
        for weather, pred in preds.items():
            predicted_weathers[weather].append(round(pred.tolist()[0][0]))
    return predicted_weathers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/config.yml")
    parser.add_argument("--result_images", default="./results/")
    args = parser.parse_args()

    os.makedirs(args.result_images, exist_ok=True)
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    img_paths: List = []
    with open(config["val_path"], "r") as f:
        val_annos = [sample.replace("\n", "") for sample in f.readlines()]
    with open(config["classes_path"], "r") as f:
        classes = f.read().rstrip().split("\n")
    img_paths = []
    gts: Dict[str, List] = {class_: [] for class_ in classes}
    for val_sample in val_annos:
        file_path, labels = val_sample.split()[0], [
            int(label) for label in val_sample.split()[1:]
        ]
        img_paths.append(os.path.join(config["val_images_root"], file_path))
        for i, class_ in enumerate(classes):
            gts[class_].append(labels[i])

    pred_weathers = predict(config, img_paths, classes)
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
                (0, 255, 0)
                if gts[class_][i] == pred_weathers[class_][i]
                else (0, 0, 255)
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
                "yes" if pred_weathers[class_][i] else "no",
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
        cv2.imwrite(os.path.join(args.result_images, img_path.split("/")[-1]), img_2)
