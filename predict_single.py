import cv2
import torch
import numpy as np
import argparse
from PIL import Image
import os
from utils.utils import get_transforms

IMG_MEANS = [0.485, 0.456, 0.406]
IMG_STDS = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# visualize prediction result
def visualize(image, predicted_classes):
    image_bgr = np.array(image)[:, :, ::-1].copy() 
    h, w, c = image_bgr.shape
    padded_region = np.ones((int(h*0.16),w,c), dtype=np.uint8) * 255
    padded_image = np.vstack([padded_region, image_bgr])
    x_start = int(w*0.16)
    scale_thickness = round(w/960,2)
    for class_, val in predicted_classes.items():
        cv2.putText(
            padded_image,
            class_,
            (x_start, int(h*0.1)),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            scale_thickness,
            (0, 255, 0) if val else (190, 190, 190),
            round(scale_thickness),
        )
        x_start += int(w*0.16)
    return padded_image

# get prediction from model
def predict_single(args):
    with open(args.classes, "r") as f:
        classes = f.read().rstrip().split("\n")
    model = torch.load(args.model_file,map_location=DEVICE)
    model.eval()
    img_ori = Image.open(args.image).convert("RGB")
    transform_pipeline = get_transforms((args.input_size,args.input_size),IMG_MEANS, IMG_STDS,train_pipeline=False)
    img = transform_pipeline(img_ori)
    img = torch.stack([img]).to(DEVICE)
    with torch.no_grad():
        preds = model(img)
    pred_classes = {class_:round(preds[class_].tolist()[0][0]) for class_ in classes}
    result_image = visualize(img_ori, pred_classes)
    saved_folder = './predicted_images'
    os.makedirs(saved_folder,exist_ok=True)
    saved_path = os.path.join(saved_folder, f"result_{os.path.basename(args.image)}") 
    cv2.imwrite(saved_path, result_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for single image prediction multi-label-binary-classification")
    parser.add_argument(
        "--image",
        default="./test_images/test1.jpg",
        help="Path of input image",
    )
    parser.add_argument(
        "--model_file",
        default="./multi-label-weather-data/models/inceptionv2_backbone/best_model.pt",
        help="Path of model file",
    )
    parser.add_argument(
        "--classes",
        default="./multi-label-weather-data/dataset/data/classes.txt",
        help="Path of classes txt file",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=299,
        help="Size of model input, (224 or 299)",
    )

    predict_single(parser.parse_args())


    
