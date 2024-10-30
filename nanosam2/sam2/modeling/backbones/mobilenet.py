import torch
from torchvision import models

class MobileNetLarge(torch.nn.Module):
    def __init__(self, pretrained=True) -> None:
        super(MobileNetLarge, self).__init__()
        self.base = models.mobilenet_v3_large(pretrained).features

    def forward(self, x):
        feature_maps = []
        x = self.base[0](x)
        x = self.base[1](x)
        x = self.base[2](x)
        x = self.base[3](x)
        feature_maps.append(x)
        x = self.base[4](x)
        x = self.base[5](x)
        x = self.base[6](x)
        feature_maps.append(x)
        x = self.base[7](x)
        x = self.base[8](x)
        x = self.base[9](x)
        x = self.base[10](x)
        x = self.base[11](x)
        x = self.base[12](x)
        feature_maps.append(x)
        x = self.base[13](x)
        x = self.base[14](x)
        x = self.base[15](x)
        x = self.base[16](x)
        feature_maps.append(x)
        return feature_maps
    

def mobilenet_v3_small():
    return MobileNetLarge("v3_small")

if __name__ == "__main__":
    model = MobileNetLarge()
    print(model)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    for i in range(len(y)):
        print(f"Feature map {i+1} shape: {y[i].shape}")
    
