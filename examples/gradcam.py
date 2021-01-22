import argparse

import cv2
import numpy as np
from torchvision import models

from cvtoolkit.gradcam import GradCAM
from examples.utils import preprocess_images, show_cam_on_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--input-path', type=str, default='./data/water-bird.jpeg',
                        help='Input image path')
    parser.add_argument('--output-path', type=str, default='./data/water-bird_gradcam.jpeg',
                        help='Output image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = models.resnet50(pretrained=True)
    gradcam = GradCAM(model, target_module=model.layer4[2])

    img1 = cv2.imread("data/cat_dog.png", 1)
    img1 = np.float32(cv2.resize(img1, (448, 448))) / 255
    img1 = img1[:, :, ::-1]

    img2 = cv2.imread("data/water-bird.jpeg", 1)
    img2 = np.float32(cv2.resize(img2, (448, 448))) / 255
    img2 = img2[:, :, ::-1]

    images = np.array([img1, img2])
    input_images = preprocess_images(images)

    batch_cams = gradcam(input_images)
    for i, cams in enumerate(batch_cams):
        img = images[i]
        for c in cams:
            cam = show_cam_on_image(img, cams[c])
            cv2.imwrite(f"data/gradcam_{i}.jpg", cam)
