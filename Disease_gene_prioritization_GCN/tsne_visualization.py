# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:03:13 2023

@author: indra
"""
import numpy as np
matrix = np.load('prediction.npy')
def tsne_visualization(matrix):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    plt.figure(dpi=300)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=0,
            n_iter=1000)
    tsne_results = tsne.fit_transform(matrix)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('.\VisualPlots.png')
    
tsne_visualization(matrix)