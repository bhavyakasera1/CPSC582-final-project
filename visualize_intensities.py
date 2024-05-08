import matplotlib.pyplot as plt
import shutil
import os
import random
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import seaborn as sns
import pandas as pd


def collate_lgg_data(dataset_path):
    for patient in os.listdir(dataset_path):
        if 'TCGA' in patient:
            for image in os.listdir(os.path.join(dataset_path, patient)):
                if 'mask' not in image:
                    print("image: ", image)
                    shutil.copy(os.path.join(dataset_path, patient, image), os.path.join("data/LGG_all", "all_mris", image))

def collate_br35h_data(dataset_path):
    for folder in os.listdir(dataset_path):
        if 'yes' in folder or 'no' in folder:
            for image in os.listdir(os.path.join(dataset_path, folder)):
                print("image: ", image)
                shutil.copy(os.path.join(dataset_path, folder, image), os.path.join("data/BR35H_all", "all_mris", image))

scaler = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()

def plot_intensities(lgg_image_folder, br35h_image_folder):
    lgg_intensities = [] 
    for image in os.listdir(lgg_image_folder):
        image_path = os.path.join(lgg_image_folder, image)
        img = np.array(Image.open(image_path))
        lgg_intensities.append(int(np.mean(img[:,:,1])))
    # sns.displot(data=lgg_intensities, kind="kde", fill=True, color="skyblue")

    br35h_intensities = []
    for image in os.listdir(br35h_image_folder):
        image_path = os.path.join(br35h_image_folder, image)
        img = np.array(Image.open(image_path))
        br35h_intensities.append(int(np.mean(img)))
    # sns.displot(data=br35h_intensities, kind="kde", fill=True, color="plum")

    df = pd.DataFrame(columns=["intensity", "dataset"])
    df["intensity"] = lgg_intensities + br35h_intensities
    df["dataset"] = ["LGG"]*len(lgg_intensities) + ["BR35H"]*len(br35h_intensities)
    sns.displot(x="intensity", data=df, kind="kde", fill=True, hue="dataset", palette=["skyblue", "plum"])
    plt.xlabel("Intensities")
    plt.ylabel("Count")
    plt.title("Intensity distributions for LGG, Br35H datasets")
    plt.savefig("plots/both_intensity_dist.png")



if __name__ == "__main__":
    plot_intensities("data/LGG_all/all_mris", "data/BR35H_all/all_mris")
    
