import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

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


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0, distance_type='euclidean'):
        super(ContrastiveLoss, self).__init__()

        self.device = "cuda"
        self.model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0,
                                                                                                 2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                             clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                             clip_preprocess.transforms[4:])  # + skip convert PIL to tensor
        self.margin = margin
        self.cos = nn.CosineSimilarity()
        self.distance_type = distance_type

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

    def clip_contrastive_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):
        source_features = self.get_text_features(source_class, norm=True)
        target_features = self.get_text_features(target_class, norm=True)

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        if self.distance_type == 'euclidean':
            euclidean_distance_near = F.pairwise_distance(target_encoding, target_features.detach(), keepdim=True)
            euclidean_distance_far_text = F.pairwise_distance(target_encoding, source_features.detach(), keepdim=True)
            euclidean_distance_far_img = F.pairwise_distance(target_encoding, src_encoding.detach(), keepdim=True)

            loss_contrastive = torch.mean(torch.pow(euclidean_distance_near, 2) +
                                          torch.pow(torch.clamp(self.margin - euclidean_distance_far_text, min=0.0), 2) + 
                                          torch.pow(torch.clamp(self.margin - euclidean_distance_far_img, min=0.0), 2))
        elif self.distance_type == 'cosine':
            consine_distance_near = self.cos(target_encoding, target_features.detach())
            consine_distance_far_text = self.cos(target_encoding, source_features.detach())
            consine_distance_far_img = self.cos(target_encoding, src_encoding.detach())

            loss_contrastive = torch.mean(torch.pow(consine_distance_near, 2) +
                                          torch.pow(torch.clamp(self.margin - consine_distance_far_text, min=0.0),
                                                    2) +
                                          torch.pow(torch.clamp(self.margin - consine_distance_far_img, min=0.0), 2))
        elif self.distance_type == 'infornce':
            euclidean_distance_near = F.pairwise_distance(target_encoding, target_features.detach(), keepdim=True)
            euclidean_distance_far_text = F.pairwise_distance(target_encoding, source_features.detach(), keepdim=True)
            euclidean_distance_far_img = F.pairwise_distance(target_encoding, src_encoding.detach(), keepdim=True)
            temperature = 1.0

            pos = torch.exp(euclidean_distance_near / temperature)
            neg_text = torch.exp(euclidean_distance_far_text / temperature)
            neg_img = torch.exp(euclidean_distance_far_img / temperature)

            loss_contrastive = torch.mean(- torch.log(pos / (pos + neg_text + neg_img)))

            #print ("pos: ", pos)
            #print ("neg_text: ", neg_text)
            #print ("neg_img:", neg_img)

            #print ("loss_contrastive: ", loss_contrastive)

        return loss_contrastive

    def forward(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):
        loss = self.clip_contrastive_loss(src_img, source_class, target_img, target_class)

        return loss