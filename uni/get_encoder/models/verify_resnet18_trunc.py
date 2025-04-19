from resnet18_trunc import resnet18_trunc_imagenet
import torch

model = resnet18_trunc_imagenet()
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(output.shape)  # Should be (1, 256)
