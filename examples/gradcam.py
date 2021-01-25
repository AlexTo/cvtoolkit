import argparse
import numpy as np
import cv2
from torchvision import models, transforms

from cvtoolkit.gradcam import GradCAM
from cvtoolkit.utils import preprocess_image, show_cam_on_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--input-path', type=str, default='./data/images/cat_dog.png',
                        help='Input image path')
    parser.add_argument('--output-path', type=str, default='./output/cat_dog_gradcam.png',
                        help='Output image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = models.resnet50(pretrained=True)
    gradcam = GradCAM(model, target_module=model.layer4[2], use_cuda=args.use_cuda)
    img = cv2.imread(args.input_path, 1)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)

    cams = gradcam(input_img)
    for c in cams:
        cam = show_cam_on_image(img, cams[c])
        cv2.imwrite(args.output_path, cam)
