import torch
import torch.nn as nn
from torchvision.models import vgg16

__all__ = ['VGGPerceptualLoss']


# VGG loss, Cite from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(vgg16(pretrained=True).features[:4].eval())
        blocks.append(vgg16(pretrained=True).features[4:9].eval())
        # 4.21 can wang
        blocks.append(vgg16(pretrained=True).features[9:16].eval())
        # 4.19 can wang
        blocks.append(vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        #print ("=============")
        #print (input.min())
        #print ("=============")
        #print (target.min())

        # normalize [-1, 1] to [0, 1] first
        #input = (input + 1) / 2
        #target = (target + 1) / 2

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i == 2:
            #if i == 1:
            #if i in feature_layers: # for 421_0
                loss += torch.nn.functional.l1_loss(x, y)
        return loss


if __name__ == '__main__':
    model = vgg16(pretrained=True)
    print(":4", model.features[:4])
    print("4:9", model.features[4:9])
    print("9:16", model.features[9:16])
    print("16:23", model.features[16:23])