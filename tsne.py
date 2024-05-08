import os
import pdb
import numpy as np
from scipy import io
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time
import glob

start_time = time.time()

relative_path = 'datasets/resnet_features_complete_office31/'

all_npys = glob.glob(os.path.dirname(os.path.realpath(__file__))+'/'+relative_path+"**/*.npy" , recursive=True)

num_plot_classes = 31
all_features_lgg = np.zeros((3929,50*53*53))
domain_names =[]
class_names = []
counter = 0

print("------- LGG -------")
for i, npy_file in enumerate(os.listdir("features_adapted/lgg_features")):
    if npy_file.endswith(".npy"):
        features = np.load("features_adapted/lgg_features/"+npy_file)
        all_features_lgg[counter, :] = features
        counter += 1

tsne1 = TSNE(n_components=3, n_jobs=16)
embeddings_lgg = tsne1.fit_transform(all_features_lgg)
vis_x_lgg = embeddings_lgg[:, 0]
vis_y_lgg = embeddings_lgg[:, 1]
vis_z_lgg = embeddings_lgg[:, 2]

all_features_br35h = np.zeros((3000,50*53*53))
domain_names =[]
class_names = []
counter = 0

print("------- Br35H -------")
for i, npy_file in enumerate(os.listdir("features_adapted/br35h_features")):
    if npy_file.endswith(".npy"):
        features = np.load("features_adapted/br35h_features/"+npy_file)
        all_features_br35h[counter, :] = features
        counter += 1

tsne2 = TSNE(n_components=3, n_jobs=16)
embeddings_br35h = tsne2.fit_transform(all_features_br35h)
vis_x_br35h = embeddings_br35h[:, 0]
vis_y_br35h = embeddings_br35h[:, 1]
vis_z_br35h = embeddings_br35h[:, 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xs=vis_x_lgg, ys=vis_y_lgg, zs=vis_z_lgg, marker='*', color="plum")
ax.scatter(xs=vis_x_br35h, ys=vis_y_br35h, zs=vis_z_br35h, marker='*', color="darkseagreen")

# plt.tight_layout()
# plt.savefig(f"plots/both_tsne.png")
plt.show()
