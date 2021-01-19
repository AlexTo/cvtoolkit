import cv2
import torch
import numpy as np

from cvtoolkit.cam import CAM


class GradCAM(CAM):
    def __init__(self, model, target_module, use_cuda=True):
        super(GradCAM, self).__init__(model, target_module, use_cuda)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activation = output

    def forward(self, img, target_categories=None):
        cams = {}
        if self.use_cuda:
            img = img.cuda()

        logits = self.model(img)
        if target_categories is None:
            target_categories = [torch.argmax(logits)]

        A = self.activation

        for target_category in target_categories:
            one_hot = torch.zeros_like(logits)
            one_hot[0][target_category] = 1
            one_hot.requires_grad_()

            yc = torch.sum(one_hot * logits)
            self.model.zero_grad()
            yc.backward(retain_graph=True)
            gradient = self.gradient
            b, k, _, _ = gradient.size()

            wc = torch.mean(gradient, axis=(2, 3))
            cam = torch.relu(torch.sum(wc.view(b, k, 1, 1) * A, dim=1)).squeeze(0)
            cam = cam.detach().cpu().numpy()
            cam = cv2.resize(cam, (img.shape[3], img.shape[2]))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cams[target_category] = cam
        return cams

    def __call__(self, img, target_categories=None):
        return super(GradCAM, self).__call__(img, target_categories)
