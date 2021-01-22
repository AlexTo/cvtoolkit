import cv2
import torch
import numpy as np

from cvtoolkit.cam import CAM


class GradCAM(CAM):
    def __init__(self, model, target_module, use_cuda=True):
        super(GradCAM, self).__init__(model, target_module, use_cuda)

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

                yc = torch.sum(one_hot * logits[i])
                self.model.zero_grad()
                yc.backward(retain_graph=True)
                gradient = self.gradients[i]
                k, _, _ = gradient.size()

                wc = torch.mean(gradient, axis=(1, 2))
                cam = torch.relu(torch.sum(wc.view(k, 1, 1) * A, dim=0))
                cam = cam.detach().cpu().numpy()
                cam = cv2.resize(cam, (images.shape[3], images.shape[2]))
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)
                cams[target] = cam
            batch_cams.append(cams)
        return batch_cams

    def __call__(self, img, target_categories=None):
        return super(GradCAM, self).__call__(img, target_categories)
