import cv2
import torch
import numpy as np
from cvtoolkit.cam import CAM


class GradCAMPlusPlus(CAM):

    def __init__(self, model, target_module, use_cuda=True):
        super(GradCAMPlusPlus, self).__init__(model, target_module, use_cuda)

    def forward(self, images, batch_targets=None):
        batch_cams = []
        if self.use_cuda:
            images = images.cuda()

        logits = self.model(images)
        batch, num_classes = logits.shape

        if batch_targets is None:
            batch_targets = torch.argmax(logits, dim=1, keepdim=True)

        for i in range(batch):
            cams = {}
            targets = batch_targets[i]

            A = self.activations[i]

            for target in targets:
                one_hot = torch.zeros_like(logits[i])
                one_hot[target] = 1
                one_hot.requires_grad_()

                sc = torch.sum(one_hot * logits[i])
                self.model.zero_grad()
                sc.backward(retain_graph=True)
                gradients = self.gradients[i]
                k, _, _ = gradients.size()

                numerator = gradients.pow(2)
                denominator = 2 * gradients.pow(2) + A.sum((1, 2)).view(k, 1, 1) * gradients.pow(3)

                alpha = numerator / denominator

                # if use exponential function
                # d_Yc/d_Akij = d_exp(sc)/d_Akij
                #             = exp(sc) * d_sc/d_Akij
                #             = exp(sc) * gradient

                wc = (alpha * torch.relu(sc.exp() * gradients)).sum((1, 2)).view(k, 1, 1)
                cam = torch.relu(torch.sum(wc.view(k, 1, 1) * A, dim=0)).squeeze(0)
                cam = cam.detach().cpu().numpy()
                cam = cv2.resize(cam, (images.shape[3], images.shape[2]))
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)
                cams[target] = cam
            batch_cams.append(cams)

        return batch_cams

    def __call__(self, img, target_categories=None):
        return super(GradCAMPlusPlus, self).__call__(img, target_categories)
