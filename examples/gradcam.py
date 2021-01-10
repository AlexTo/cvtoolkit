import argparse
import numpy as np
import cv2
from torchvision import models, transforms

from cvtoolkit.gradcam import GradCAM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--input-path', type=str, default='./data/cat_dog.png',
                        help='Input image path')
    parser.add_argument('--output-path', type=str, default='./data/cat_dog_gradcam.png',
                        help='Output image path')
    return parser.parse_args()


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    args = get_args()

    model = models.resnet50(pretrained=True)
    gradcam = GradCAM(model, target_module=model.layer4, target_layer=model.layer4[2])
    img = cv2.imread(args.input_path, 1)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)

    cam = gradcam(input_img)
    cam = show_cam_on_image(img, cam)
    cv2.imwrite(args.output_path, cam)
