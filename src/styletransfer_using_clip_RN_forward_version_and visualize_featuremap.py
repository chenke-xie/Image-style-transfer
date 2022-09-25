# -*- coding: utf-8 -*-
"""# import CLIP models"""

import numpy as np
import torch
from pkg_resources import packaging

print("Torch version:", torch.__version__)


import clip

clip.available_models()

"""# Load image data"""

import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
image_dir = os.getcwd() +"/"

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


from torchvision import transforms

from PIL import Image
from collections import OrderedDict
from pylab import *



# Select device to work on.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### load images, ordered as [style_image, content_image]
img_dirs = [image_dir, image_dir]
img_names = ['vangogh_starry_night.jpg', 'Tuebingen_Neckarfront.jpg']
imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]

for img in imgs:
  imshow(img);show()

"""# Load RN50 from CLIP"""

import clip
model_rn50, preprocess = clip.load('RN50', device=device)
for param in model_rn50.parameters():
    param.requires_grad = False

model_rn50

preprocess

"""## Modify the RN50"""

#use only vision portion to extract feature maps; delete linear layers
model_children = list(model_rn50.children())
vision_portion = model_children[0]
vision_portion = nn.Sequential(*list(vision_portion.children())[:-1])

vision_portion

"""# Preprocess the image data"""

prep = transforms.Compose([transforms.Resize(512),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul(255)),
                          ])

postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

##### load images, ordered as [style_image, content_image]
imgs_torch = [prep(img) for img in imgs]

if torch.cuda.is_available():
    imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
else:
    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch

opt_img = Variable(content_image.data.clone(), requires_grad=True)

"""# Define gram matrix and loss"""

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        #b=batch, c=channel, h=height, w=weight
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        #loss = nn.MSELoss()
        #output = loss(input, target)
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

"""# Extract feature maps from RN50"""

vision_layers = []
counti = 0
countj = 0
for i in range (len(vision_portion)):
  if type(vision_portion[i]) != nn.Sequential:
    counti += 1
    # print("layer:" + str(i))
    # print(str(model_children[i]))
    # model_weights.append(model_children[i].weight.float())
    vision_layers.append(vision_portion[i].float())
  else:
    for j in range(len(vision_portion[i])):
      countj +=1
      # print("layer Sequential:" + str(i) + "," + str(j))
      # print(str(model_children[i][j]))
      # model_weights.append(model_children[i][j].weight.float())
      vision_layers.append(vision_portion[i][j].float())

# Definde a function to pass the image through all the layers

def vision_forward(img, vision_model, layers):

  results = [vision_model[0](img)]
  for i in range(1, len(vision_model)):
    # pass the result from the last layer to the next layer
    results.append(vision_model[i](results[-1]))
  # make a copy of the `results`
  outputs = results
  outputs = [outputs[index] for index in layers]

  return outputs

"""# Define layers, loss functions, weights and compute optimization targets"""

#define layers, loss functions, weights and compute optimization targets

#style layers: 2, 5, 12, 16, 22 / content layer: 12
style_layers = [2, 5, 12, 16, 22]
content_layers = [12]
loss_layers = style_layers + content_layers
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

#weights settings
style_weights = [1e3/n**2 for n in [32, 32, 256, 512, 1024]]
content_weights = [1e0]
weights = style_weights + content_weights

#compute feature maps
style_feature_maps = vision_forward(style_image, vision_layers, style_layers)
content_feature_maps = vision_forward(content_image, vision_layers, content_layers)

#output style and content feature maps' shape
print('Feature Maps of style image:')
for i in range(len(style_feature_maps)):
  print(f"Layer {i}:", style_feature_maps[i].shape)

print('Feature Maps of content image:')
for i in range(len(content_feature_maps)):
  print(f"Layer {i}:", content_feature_maps[i].shape)

#compute optimization targets
style_targets = [GramMatrix()(A).detach() for A in style_feature_maps]
content_targets = [A.detach() for A in content_feature_maps]
targets = style_targets + content_targets

#opt_img = Variable(content_image.data.clone(), requires_grad=True)
out = vision_forward(opt_img, vision_layers, loss_layers)
layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
layer_losses

"""# Run style transfer"""

#run style transfer
max_iter = 1500
show_iter = 50

optimizer = optim.LBFGS([opt_img]);
n_iter=[0]

while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        out = vision_forward(opt_img, vision_layers, loss_layers)

        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
        
        loss = 0
        for layer_loss in layer_losses:
          loss = loss + layer_loss
        loss.backward()
        n_iter[0]+=1
        if n_iter[0]%show_iter == (show_iter-1):
            print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
        return loss
    
    optimizer.step(closure)
    
#display result
out_img = postp(opt_img.data[0].cpu().squeeze())
imshow(out_img)
gcf().set_size_inches(10,10)



# Appendix

## Visualize feature maps of content image and style image using RN50


# visualize 32 features from each layer 
# (although there are more feature maps in the upper layers)
outputs_rn50_content = vision_forward(content_image, vision_layers, range(26))
for num_layer in range(len(outputs_rn50_content)):
    plt.figure(figsize=(30, 30)) 
    layer_viz = outputs_rn50_content[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 32: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(4, 8, i + 1)
        plt.imshow(filter.cpu(), cmap='gray')
        plt.axis("off")
    print(f"Loading layer {num_layer} feature maps...")
    
    plt.show()
    plt.close()

# visualize 32 features from each layer 
# (although there are more feature maps in the upper layers)
outputs_rn50_style = vision_forward(style_image, vision_layers, range(26))
for num_layer in range(len(outputs_rn50_style)):
    plt.figure(figsize=(30, 30)) 
    layer_viz = outputs_rn50_style[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 32: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(4, 8, i + 1)
        plt.imshow(filter.cpu(), cmap='gray')
        plt.axis("off")
    print(f"Loading layer {num_layer} feature maps...")
    
    plt.show()
    plt.close()

