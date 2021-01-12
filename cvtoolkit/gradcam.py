import cv2
import torch
import numpy as np


class GradCAM:
    def __init__(self, model, target_module, use_cuda=True):
        self.model = model
        self.use_cuda = use_cuda
        self.model.eval()
        if use_cuda:
            self.model = model.cuda()

        for module in self.model.modules():
            if module == target_module:
                module.register_forward_hook(self.forward_hook)
                module.register_backward_hook(self.backward_hook)

        self.target_module = target_module
        self.saved_gradient = None

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

            one_hot = torch.sum(one_hot * logits)
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
            weights = torch.mean(self.gradient, axis=(2, 3))
            cam = torch.relu(torch.sum(weights[:, :, None, None] * A, dim=1)).squeeze(0)
            cam = cam.detach().cpu().numpy()
            cam = cv2.resize(cam, img.shape[2:])
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cams[target_category] = cam
        return cams

    def __call__(self, img, target_categories=None):
        return self.forward(img, target_categories)
