'''
Description: 
Author: Rigel Ma
Date: 2024-04-23 09:47:43
LastEditors: Rigel Ma
LastEditTime: 2024-04-26 14:37:19
FilePath: analysis.py
'''
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.ticker as ticker

def fmt(x):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${}e{{{}}}$'.format(a, b)

def embedding_tsne(embed, save_name='embed', layer=False):
    user_tsneData = TSNE().fit_transform(embed.detach().cpu().numpy())
    x, y = user_tsneData[:,0],user_tsneData[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    plt.ticklabel_format(style='plain', axis='x')
    plt.ticklabel_format(style='plain', axis='y')

    ax_sca = ax.scatter(x, y,c=z, s=20,cmap='Spectral') 
    ax.set_title(f'{save_name}', fontdict={'fontsize':14})
    ax.set_xlabel('features', fontdict={'fontsize': 18})
    ax.set_ylabel('features', fontdict={'fontsize': 18})

    fig.colorbar(ax_sca, ax=ax, orientation='vertical', format=ticker.FuncFormatter(fmt))

    fig.savefig(f'{save_name}.png', dpi=400)

