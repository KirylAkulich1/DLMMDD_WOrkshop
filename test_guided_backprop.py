import torch
import torch.nn as nn

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.forward_relu_outputs = []
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.forward_relu_outputs.append(output)
            
        def backward_hook(module, grad_input, grad_output):
            forward_output = self.forward_relu_outputs.pop()
            positive_mask = (forward_output > 0).float()
            positive_grad_output = torch.clamp(grad_output[0], min=0.0)
            return (positive_mask * positive_grad_output,)
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
    
    def generate_gradients(self, input_image, target_class):
        self.model.eval()
        input_image.requires_grad = True
        output = self.model(input_image)
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output)
        gradients = input_image.grad.data
        return gradients

model = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.ReLU(inplace=True), # Try inplace=True to see if it triggers
    nn.Conv2d(16, 16, 3),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(16, 10)
)

gb = GuidedBackprop(model)
img = torch.randn(1, 3, 32, 32)
grad = gb.generate_gradients(img, 5)
print(grad.sum())
