# LGG preprocess

import shutil
import os
import random
import numpy as np
from PIL import Image

def preprocess_lgg():
    for folder in os.listdir('data/LGG/lgg-mri-segmentation'):
        if "TCGA" in folder:
            for mri in os.listdir(f'data/LGG/lgg-mri-segmentation/{folder}'):
                if 'mask' not in mri:
                    mask = np.array(Image.open(f'data/LGG/lgg-mri-segmentation/{folder}/{mri.split(".tif")[0]}_mask.tif'))
                    if np.sum(mask) == 0:
                        shutil.copy(f'data/LGG/lgg-mri-segmentation/{folder}/{mri}', f'data/LGG_preprocessed/normal/{mri}')
                    else:
                        shutil.copy(f'data/LGG/lgg-mri-segmentation/{folder}/{mri}', f'data/LGG_preprocessed/tumor/{mri}')

def preprocess_br35h():
    for mri in os.listdir('data/Br35H/no'):
        shutil.copy(f'data/Br35H/no/{mri}', f'data/Br35H_preprocessed/normal/{mri}')
    for mri in os.listdir('data/Br35H/yes'):
        shutil.copy(f'data/Br35H/yes/{mri}', f'data/Br35H_preprocessed/tumor/{mri}')

if __name__=="__main__":
    preprocess_br35h()