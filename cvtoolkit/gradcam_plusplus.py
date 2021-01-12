import cv2
import torch
import numpy as np


class GradCAMPlusPlus:
    def __init__(self, model, target_module, target_layer, use_cuda=True):
        self.model = model
        self.use_cuda = use_cuda
        self.model.eval()
        if use_cuda:
            self.model = model.cuda()

        self.target_module = target_module
        self.target_layer = target_layer
        self.saved_gradient = None

    def save_gradient(self, grad):
        self.saved_gradient = grad

    def __call__(self, img, target_category=None):
        if self.use_cuda:
            img = img.cuda()
        img.requires_grad_()

        output = img
        for name, module in self.model.named_children():
            if module == self.target_module:
                for _, sub_module in module.named_children():
                    A = sub_module(output)
                    output = A
                    if sub_module == self.target_layer:
                        output.register_hook(self.save_gradient)
            else:
                output = module(output)

            if "avgpool" in name.lower():
                output = output.view(output.size(0), -1)

        if target_category is None:
            target_category = torch.argmax(output)

        one_hot = torch.zeros_like(output)
        one_hot[0][target_category] = 1
        one_hot.requires_grad_()

        self.model.zero_grad()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        weights = torch.mean(self.saved_gradient, axis=(2, 3))

        cam = torch.relu(torch.sum(weights[:, :, None, None] * A, dim=1)).squeeze(0)
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
