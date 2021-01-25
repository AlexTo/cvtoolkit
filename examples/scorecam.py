import argparse
import numpy as np
import cv2
from torchvision import models

from cvtoolkit.scorecam import ScoreCAM
from examples.gradcam import show_cam_on_image
from cvtoolkit.utils import preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--input-path', type=str, default='./data/ILSVRC2012_val_00002193.jpg',
                        help='Input image path')
    parser.add_argument('--output-path', type=str, default='./data/ILSVRC2012_val_00002193_scorecam.jpg',
                        help='Output image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = models.alexnet(pretrained=True)
    scorecam = ScoreCAM(model, target_module=model.features[10])
    img = cv2.imread(args.input_path, 1)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)

    cams = scorecam(input_img)
    for c in cams:
        cam = show_cam_on_image(img, cams[c])
        cv2.imwrite(args.output_path, cam)
