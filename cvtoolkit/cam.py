from abc import abstractmethod


class CAM:
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
        self.gradients = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations = output

    @abstractmethod
    def forward(self, img, target_categories=None):
        pass

    @abstractmethod
    def __call__(self, img, target_categories=None):
        return self.forward(img, target_categories)
