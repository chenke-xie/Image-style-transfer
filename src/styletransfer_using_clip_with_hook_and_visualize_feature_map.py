# -*- coding: utf-8 -*-


import numpy as np
import torch
from pkg_resources import packaging

print("Torch version:", torch.__version__)

import clip

clip.available_models()

"""# Load image data"""

import time
import os 
image_dir = os.getcwd() +"/"

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict
from pylab import *

# Select device to work on.
device = torch.device("cpu")

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

model_rn50.visual

preprocess


"""# Preprocess the image data"""

prep = transforms.Compose([transforms.Resize(224),
                           transforms.CenterCrop(size=(224, 224)),
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

# if torch.cuda.is_available():
#     imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
# else:
imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch

opt_img = Variable(content_image.data.clone(), requires_grad=True)

style_activations = {}
content_activations = {}
opt_activations = {}
def getActivation(name):
  def hook(model, input, output):
    style_activations[name] = output.detach()
    content_activations[name] = output.detach()
    opt_activations[name] = output
  return hook

h0 = model_rn50.visual.conv1.register_forward_hook(getActivation('conv1'))
h1 = model_rn50.visual.bn1.register_forward_hook(getActivation('bn1'))
h2 = model_rn50.visual.relu1.register_forward_hook(getActivation('relu1'))
h3 = model_rn50.visual.conv2.register_forward_hook(getActivation('conv2'))
h4 = model_rn50.visual.bn2.register_forward_hook(getActivation('bn2'))
h5 = model_rn50.visual.relu2.register_forward_hook(getActivation('relu2'))
h6 = model_rn50.visual.conv3.register_forward_hook(getActivation('conv3'))
h7 = model_rn50.visual.bn3.register_forward_hook(getActivation('bn3'))
h8 = model_rn50.visual.relu3.register_forward_hook(getActivation('relu3'))
h9 = model_rn50.visual.avgpool.register_forward_hook(getActivation('avgpool'))

L1_Bottleneck_0 = model_rn50.visual.layer1[0].register_forward_hook(getActivation('L1_Bottleneck_0'))
L1_Bottleneck_1 = model_rn50.visual.layer1[1].register_forward_hook(getActivation('L1_Bottleneck_1'))
L1_Bottleneck_2 = model_rn50.visual.layer1[2].register_forward_hook(getActivation('L1_Bottleneck_2'))


L2_Bottleneck_0 = model_rn50.visual.layer2[0].register_forward_hook(getActivation('L2_Bottleneck_0'))
L2_Bottleneck_1 = model_rn50.visual.layer2[1].register_forward_hook(getActivation('L2_Bottleneck_1'))
L2_Bottleneck_2 = model_rn50.visual.layer2[2].register_forward_hook(getActivation('L2_Bottleneck_2'))
L2_Bottleneck_3 = model_rn50.visual.layer2[3].register_forward_hook(getActivation('L1_Bottleneck_3'))


L3_Bottleneck_0 = model_rn50.visual.layer3[0].register_forward_hook(getActivation('L3_Bottleneck_0'))
L3_Bottleneck_1 = model_rn50.visual.layer3[1].register_forward_hook(getActivation('L3_Bottleneck_1'))
L3_Bottleneck_2 = model_rn50.visual.layer3[2].register_forward_hook(getActivation('L3_Bottleneck_2'))
L3_Bottleneck_3 = model_rn50.visual.layer3[3].register_forward_hook(getActivation('L3_Bottleneck_3'))
L3_Bottleneck_4 = model_rn50.visual.layer3[4].register_forward_hook(getActivation('L3_Bottleneck_4'))
L3_Bottleneck_5 = model_rn50.visual.layer3[5].register_forward_hook(getActivation('L3_Bottleneck_5'))


L4_Bottleneck_0 = model_rn50.visual.layer4[0].register_forward_hook(getActivation('L4_Bottleneck_0'))
L4_Bottleneck_1 = model_rn50.visual.layer4[1].register_forward_hook(getActivation('L4_Bottleneck_1'))
L4_Bottleneck_2 = model_rn50.visual.layer4[2].register_forward_hook(getActivation('L4_Bottleneck_2'))

test = model_rn50.visual(style_image)
test = style_activations['conv1']
test.shape

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
        #等价于表达式：
        #loss = nn.MSELoss()
        #output = loss(input, target)
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)



"""# Define layers, loss functions, weights and compute optimization targets"""

#define layers, loss functions, weights and compute optimization targets

#style layers: 2, 5, 12, 16, 22 / content layer: 12
style_layers = ['bn1', 'bn2', 'L1_Bottleneck_1', 'L2_Bottleneck_2', 'L3_Bottleneck_4']
content_layers = ['L1_Bottleneck_1']
loss_layers = style_layers + content_layers
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

#weights settings
style_weights = [1e3/n**2 for n in [32, 32, 256, 512, 1024]]
content_weights = [1e0]
weights = style_weights + content_weights
print(weights)

#get activation outputs
style_result = model_rn50.visual(style_image)
style_activations_list = []
for i, (k, v) in enumerate(style_activations.items()):
  if k in style_layers:
    style_activations_list.append(v)

content_result = model_rn50.visual(content_image)
content_activations_list = []
for i, (k, v) in enumerate(content_activations.items()):
  if k in content_layers:
    content_activations_list.append(v)


#output style and content feature maps' shape
print('Feature Maps of style image:')
for i in range(len(style_activations_list)):
  
  print(f"Layer {i}:", style_activations_list[i].shape)

print('Feature Maps of content image:')
for i in range(len(content_activations_list)):
  print(f"Layer {i}:", content_activations_list[i].shape)

#compute optimization targets
style_targets = [GramMatrix()(A).detach() for A in style_activations_list]
content_targets = [A.detach() for A in content_activations_list]
targets = style_targets + content_targets

#opt_img = Variable(content_image.data.clone(), requires_grad=True)
out_img_result = model_rn50.visual(opt_img)
opt_activations_list = []
for i, (k, v) in enumerate(opt_activations.items()):
  if k in style_layers:
    opt_activations_list.append(v)
for i, (k, v) in enumerate(opt_activations.items()):
  if k in content_layers:
    opt_activations_list.append(v)
  

layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(opt_activations_list)]
layer_losses

"""# Run style transfer"""

#run style transfer
max_iter = 500
show_iter = 50

optimizer = optim.LBFGS([opt_img]);
n_iter=[0]

while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        out = model_rn50.visual(opt_img)
        opt_activations_list = []
        for i, (k, v) in enumerate(opt_activations.items()):
          if k in style_layers:
            opt_activations_list.append(v)
        for i, (k, v) in enumerate(opt_activations.items()):
          if k in content_layers:
            opt_activations_list.append(v)

        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(opt_activations_list)]
        
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


## Visualize feature maps of content image and style image using RN50


# visualize 32 features from each layer 
# (although there are more feature maps in the upper layers)
# outputs_rn50_content = model_rn50.visual(opt_img)

for num_layer in range(len(content_activations_list)):
    plt.figure(figsize=(30, 30)) 
    # print(outputs_rn50_content[num_layer].shape)
    layer_viz = content_activations_list[num_layer][0, :, :, :]
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
# outputs_rn50_style = model_rn50.visual(opt_img)
for num_layer in range(len(style_activations_list)):
    plt.figure(figsize=(30, 30)) 
    layer_viz = style_activations_list[num_layer][0, :, :, :]
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

