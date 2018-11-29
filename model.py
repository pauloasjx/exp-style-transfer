import torch
import torch.optim as optim

from torchvision import models

from tensor import gram_matrix
from image import im_convert


class Model():
    def __init__(self, device=None):
        self.device = device

        vgg = models.vgg19(pretrained=True).features

        for param in vgg.parameters():
            param.requires_grad_(False)

        self.model = vgg.to(self.device)

    def get_features(self, image):
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}

        features = {}
        x = image

        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

    def optimize_style(self, content, style, style_weights, content_weight, style_weight, steps):
        content_features = self.get_features(content)
        style_features = self.get_features(style)

        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

        target = content.clone().requires_grad_(True).to(self.device)

        optimizer = optim.Adam([target], lr=0.003)

        for ii in range(1, steps + 1):
            target_features = self.get_features(target)

            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

            style_loss = 0

            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                _, d, h, w = target_feature.shape

                style_gram = style_grams[layer]

                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

                style_loss += layer_style_loss / (d * h * w)

            total_loss = content_weight * content_loss + style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return im_convert(target)
