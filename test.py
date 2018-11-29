import matplotlib.pyplot as plt

import torch

from image import load_image
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content = load_image('images/octopus.jpg').to(device)
style = load_image('images/hockney.jpg', shape=content.shape[-2:]).to(device)

model = Model(device)

style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1
style_weight = 1e6

image = model.optimize_style(content, style, style_weights, content_weight, style_weight, 10)

plt.imshow(image)
