import torch
import torch.nn.functional as F

from cvtoolkit.cam import CAM


class ScoreCAM(CAM):
    def __init__(self, model, target_module, use_cuda=True):
        super(ScoreCAM, self).__init__(model, target_module, use_cuda)

    def forward(self, x, targets=None):
        with torch.no_grad():
            if self.use_cuda:
                x = x.cuda()

            b, c, h, w = x.size()
            logits = self.model(x)
            if targets is None:
                targets = torch.argmax(logits, dim=1, keepdim=True)

            A = self.activation
            b, k, u, v = A.size()

            saliency_maps = F.interpolate(A, size=(h, w), mode="bilinear", align_corners=False).view(-1, h * w)
            saliency_map_max = saliency_maps.max(dim=1).values
            saliency_map_min = saliency_maps.min(dim=1).values

            # ignore saliency maps where max values equal min values
            selected = saliency_map_max != saliency_map_min

            saliency_maps = saliency_maps[selected]
            saliency_map_max = saliency_map_max[selected].view(-1, 1)
            saliency_map_min = saliency_map_min[selected].view(-1, 1)

            norm_saliency_map = (saliency_maps - saliency_map_min) / (saliency_map_max - saliency_map_min)
            norm_saliency_map = norm_saliency_map.view(b, -1, h, w)

            _, k, _, _ = norm_saliency_map.shape
            x = x.reshape(-1)
            x = x.repeat(k)

            norm_saliency_map = norm_saliency_map.permute(1, 0, 2, 3)
            norm_saliency_map = norm_saliency_map.repeat(1, 1, c, 1)
            norm_saliency_map = norm_saliency_map.reshape(-1)

            s = (norm_saliency_map * x).reshape(-1, c, h, w)

            outputs = self.model(s)

            pass

    def __call__(self, img, targets=None):
        return self.forward(img, targets)
