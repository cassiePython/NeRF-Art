import torch
import clip
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import InterpolationMode

imagenet_templates = [
    'a bad photo of a {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


class PatchNCELoss(torch.nn.Module):

    def __init__(self,target_hw:list):
        super(PatchNCELoss, self).__init__()

        self.device = "cuda"
        self.model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0,
                                                                                                 2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                             #clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                             [transforms.Resize([224, 224])] +
                                             clip_preprocess.transforms[4:])  # + skip convert PIL to tensor
        self.cos = nn.CosineSimilarity()
        self.temperature = 0.07


        #self.ZeroPad = nn.ZeroPad2d(padding=(405, 405, 720, 720))
        #self.ZeroPad = nn.ZeroPad2d(padding=(540, 540, 960, 960))
        self.ZeroPad = nn.ZeroPad2d(padding=(675, 675, 1200, 1200))
        if target_hw[0] == target_hw[1]: #!HARDCODED Aug 25: for girl scene, which is 1:1
            self.ZeroPad = nn.ZeroPad2d(padding = (128, 128, 128, 128))
            # self.ZeroPad = nn.ZeroPad2d(padding = (256,256,256,256))
        # can 
        self.ZeroPad = nn.ZeroPad2d(padding=(270, 270, 480, 480))
        #self.ZeroPad = nn.ZeroPad2d(padding=(135, 135, 240, 240))
        # self.ZeroPad = nn.ZeroPad2d(padding=(32, 32, 32, 32)) # for llff scene
        self.resize = transforms.Resize([target_hw[0], target_hw[1]],interpolation=InterpolationMode.BICUBIC)
        print("PatchNCE target size: ",target_hw)
    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)


    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def clip_contrastive_loss(self, source_classes: list, target_img: torch.Tensor, target_class: str):
        source_feature_list = []
        for source in source_classes:
            source_feature = self.get_text_features(source, norm=True)
            source_feature_list.append(source_feature)

        target_feature = self.get_text_features(target_class, norm=True)
        target_encoding = self.get_image_features(target_img)

        consine_distance_near = self.cos(target_encoding, target_feature.detach())

        neg_texts_sum = 0
        for source_feature in source_feature_list:
            consine_distance_far_text = self.cos(target_encoding, source_feature.detach())
            neg_text = torch.exp(consine_distance_far_text / self.temperature)
            neg_texts_sum += neg_text

        pos = torch.exp(consine_distance_near / self.temperature)
        loss_contrastive = torch.mean(- torch.log(pos / (pos + neg_texts_sum)))

        return loss_contrastive

    def forward(self, source_classes: list, target_img: torch.Tensor, target_class: str, is_full_res:bool):

        #! Aug 24: temp disabled for girl dataset
        #print ("===== target_img", target_img.shape)
        target_img = self.ZeroPad(target_img)
        #print ("===== target_img", target_img.shape)
        #! Aug 24: added diferent resize for campability with is_full_res
        #! for keeping the same scale with previous results
        #! i.e when target_img is 960 height, then is_full_res == True
        target_img = self.resize(target_img)
        #print ("===== target_img", target_img.shape)


        # target_img 1, 3, W, H
        B, C, H, W = target_img.shape
        # the patch size to be cropped
        th, tw = 224, 224
        if not is_full_res:
            th, tw = 112, 112
        #th, tw = 128, 128
        total_loss = 0
        for _ in range(12):
        #for _ in range(18):
            i = torch.randint(0, H - th + 1, size=(1,)).item()
            if H != W: 
                if not is_full_res:
                    i = torch.randint(100, H - th + 1-100, size=(1,)).item()
                else:
                    i = torch.randint(200, H - th + 1-200, size=(1,)).item()
            #!HARDCODED Aug 24: for girl dataset
            else: #!HARDCODED Aug 25: for girl scene, which is 1:1
                if not is_full_res:
                    i = torch.randint(40, H - th + 1-40, size=(1,)).item()
                else:
                    i = torch.randint(80, H - th + 1-80, size=(1,)).item()
            # i = torch.randint(0, H - th + 1, size=(1,)).item()#for llff dataset
            j = torch.randint(0, W - tw + 1, size=(1,)).item()
            img = transforms.functional.crop(target_img, i, j, th, tw)
            if not is_full_res:
                #2x upsample
                img = F.interpolate(img, size=(224, 224), mode='bicubic', align_corners=False)

            loss = self.clip_contrastive_loss(source_classes, img, target_class)
            total_loss = total_loss + loss

        return total_loss
