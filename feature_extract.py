import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
# from analyse_data import make_directory
# Load the pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Use the model object to select the desired layer
# layer = model._modules.get('avgpool')
layer = model._modules.get('fc')

# Set model to evaluation mode
model.eval()

# Image transforms
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_vector(image_name):
    img = Image.open(image_name).convert('RGB')
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    my_embedding = torch.zeros((1,1000))
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    return my_embedding.numpy()

# abc = get_vector("datasets/test_office_31_data/amazon/back_pack/frame_0001.jpg")

relative_path = 'data/Br35H_all/all_mris'
all_jpgs = glob.glob(os.path.dirname(os.path.realpath(__file__))+'/'+relative_path+"**/*.jpg" , recursive=True)
for image_path in tqdm(all_jpgs):
    feat_vec = get_vector(image_path)
    path_splits = image_path.split('/')
    img_id = path_splits[-1].split('.')[0]
    save_path = 'features/br35h_features/'
    make_directory(save_path)
    with open(save_path + img_id + ".npy", 'wb') as f:
        np.save(f, feat_vec)