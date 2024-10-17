import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

class ResNet(nn.Module):
    def __init__(self, block_type, layers):
        super(ResNet, self).__init__()
        self.block = self._get_block(block_type)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

    def _get_block(self, block_type):
        if block_type == "basic":
            return BasicBlock
        elif block_type == "bottleneck":
            return Bottleneck
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.block.expansion),
            )

        layers = []
        layers.append(self.block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.block.expansion
        for _ in range(1, blocks):
            layers.append(self.block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_maps = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        feature_maps.append(x)

        x = self.layer2(x)
        feature_maps.append(x)

        x = self.layer3(x)
        feature_maps.append(x)

        x = self.layer4(x)
        feature_maps.append(x)

        return feature_maps

def resnet18():
    return ResNet("basic", [2, 2, 2, 2])

def resnet34():
    return ResNet("basic", [3, 4, 6, 3])

def resnet50():
    return ResNet("bottleneck", [3, 4, 6, 3])

if __name__ == "__main__":
    model = resnet18()
    print(model)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    for i in range(len(y)):
        print(f"Feature map {i+1} shape: {y[i].shape}")
    

