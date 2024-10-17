import torch
import torchvision.models as models

# Load the pre-trained ResNet-50 model
pretrained_resnet = models.resnet50(pretrained=True)

# Set the model to evaluation mode
pretrained_resnet.eval()

# Sample input tensor
sample_input = torch.randn(1, 3, 224, 224)  # assuming input size similar to ImageNet (3 channels, 224x224)

# Pass the sample input through the model and print the shape of the features after each layer
with torch.no_grad():
    x = sample_input
    for name, layer in pretrained_resnet.named_children():
        x = layer(x)
        print(name, "output shape:", x.shape)