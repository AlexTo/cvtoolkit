import argparse
import os
import pickle

import cv2
import numpy as np
import torch

from cvtoolkit.graphcam import gcn_resnet101
from cvtoolkit.utils import preprocess_image, show_cam_on_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', '-i', default=448, type=int,
                        metavar='N', help='image size (default: 448)')
    parser.add_argument('--embedding', default='./data/coco/coco_glove_word2vec.pkl',
                        type=str, metavar='EMB', help='path to embedding (default: glove)')
    parser.add_argument('--embedding-length', default=300, type=int, metavar='EMB',
                        help='embedding length (default: 300)')
    parser.add_argument('--checkpoint', default='./checkpoint/coco_checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--adj-path', default='./data/coco/coco_adj.pkl', type=str, metavar='ADJ',
                        help='Adj path (default: ./data/coco/coco_adj.pkl)')
    parser.add_argument('--adj-threshold', default=0.4, type=float, metavar='ADJT',
                        help='Adj threshold (default: 0.4)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--input-path', type=str, default='./data/images/cat_dog.png',
                        help='Input image path')
    parser.add_argument('--output-path', type=str, default='./output/',
                        help='Output image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    num_classes = 80

    graph_cam = gcn_resnet101(num_classes=num_classes, t=args.adj_threshold, adj_file=args.adj_path,
                              in_channel=args.embedding_length)

    with open(args.embedding, 'rb') as f:
        embedding = torch.from_numpy(pickle.load(f))

    checkpoint = torch.load(args.checkpoint)
    graph_cam.load_state_dict(checkpoint["state_dict"])
    graph_cam.eval()

    img = cv2.imread(args.input_path, 1)
    img = cv2.resize(img, (args.image_size, args.image_size))
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)

    preds, A, w = graph_cam(input_img, embedding, return_cam=True)
    _, k, _, _ = A.shape

    labels = [20, 28]  # cat, dog
    for label in labels:
        wc = w[:, label].view(k, 1, 1)
        cam = torch.relu(torch.sum(wc * A[0], dim=0)).detach().cpu().numpy()
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = show_cam_on_image(img, cam)

        filename, fileext = os.path.splitext(os.path.basename(args.input_path))

        cv2.imwrite(os.path.join(args.output_path, f"{filename}_graphcam_{label}{fileext}"), cam)
