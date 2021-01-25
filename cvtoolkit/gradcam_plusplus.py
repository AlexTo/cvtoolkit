import cv2
import torch
import numpy as np
from cvtoolkit.cam import CAM


class GradCAMPlusPlus(CAM):

    def __init__(self, model, target_module, use_cuda=True):
        super(GradCAMPlusPlus, self).__init__(model, target_module, use_cuda)

    def forward(self, img, targets=None):
        cams = {}
        if self.use_cuda:
            img = img.cuda()

        logits = self.model(img)
        if targets is None:
            targets = [torch.argmax(logits)]

        A = self.activation

        for target_category in targets:
            one_hot = torch.zeros_like(logits)
            one_hot[0][target_category] = 1
            one_hot.requires_grad_()

            sc = torch.sum(one_hot * logits)
            self.model.zero_grad()
            sc.backward(retain_graph=True)
            gradient = self.gradient
            b, k, _, _ = gradient.size()

            numerator = gradient.pow(2)
            denominator = 2 * gradient.pow(2) + A.sum((2, 3)).view(b, k, 1, 1) * gradient.pow(3)

            alpha = numerator / denominator

            # if use exponential function
            # d_Yc/d_Akij = d_exp(sc)/d_Akij
            #             = exp(sc) * d_sc/d_Akij
            #             = exp(sc) * gradient

            wc = (alpha * torch.relu(sc.exp() * gradient)).sum((2, 3)).view(b, k, 1, 1)
            cam = torch.relu(torch.sum(wc.view(b, k, 1, 1) * A, dim=1)).squeeze(0)
            cam = cam.detach().cpu().numpy()
            cam = cv2.resize(cam, (img.shape[3], img.shape[2]))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cams[target_category] = cam

        return cams

    def __call__(self, img, targets=None):
        return super(GradCAMPlusPlus, self).__call__(img, targets)
