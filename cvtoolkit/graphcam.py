import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Parameter

from cvtoolkit.utils import gen_A, gen_adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphCAM(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None, use_cuda=True):
        super(GraphCAM, self).__init__()
        self.use_cuda = use_cuda
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, img, inp, return_cam=False, targets=None):
        cams = {}
        if self.use_cuda:
            img = img.cuda()
        A = self.features(img)
        _, k, _, _ = A.shape
        feature = self.pooling(A)
        feature = feature.view(feature.size(0), -1)

        adj = gen_adj(self.A).detach()
        w = self.gc1(inp, adj)
        w = self.relu(w)
        w = self.gc2(w, adj)

        w = w.transpose(0, 1)
        logits = torch.matmul(feature, w)

        if not return_cam:
            return logits
        else:
            if targets is None:
                targets = [torch.argmax(logits)]
            for target in targets:
                wc = w[:, target].view(k, 1, 1)
                cam = torch.relu(torch.sum(wc * A[0], dim=0)).detach().cpu().numpy()
                cam = cv2.resize(cam, (img.shape[3], img.shape[2]))
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)
                cams[target] = cam

            return cams


def graph_cam(num_classes, t, pretrained=True, adj_file=None, in_channel=300, use_cuda=True):
    model = models.resnet101(pretrained=pretrained)
    return GraphCAM(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel, use_cuda=use_cuda)
