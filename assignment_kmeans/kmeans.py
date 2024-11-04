#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import itertools


def k_means(x, k, max_iter, show=False, init_means=None):
    """
    Implementation of the k-means clustering algorithm.

    :param x:               feature vectors, np array (dim, number_of_vectors)
    :param k:               required number of clusters, scalar
    :param max_iter:        stopping criterion: max. number of iterations
    :param show:            (optional) boolean switch to turn on/off visualization of partial results
    :param init_means:      (optional) initial cluster prototypes, np array (dim, k)

    :return cluster_labels: cluster index for each feature vector, np array (number_of_vectors, )
                            array contains only values from 0 to k-1,
                            i.e. cluster_labels[i] is the index of a cluster which the vector x[:,i] belongs to.
    :return centroids:      cluster centroids, np array (dim, k), same type as x
                            i.e. centroids[:,i] is the center of the i-th cluster.
    :return sq_dists:       squared distances to the nearest centroid for each feature vector,
                            np array (number_of_vectors, )

    Note 1: The iterative procedure terminates if either maximum number of iterations is reached
            or there is no change in assignment of data to the clusters.

    Note 2: DO NOT MODIFY INITIALIZATIONS

    """
    # Number of vectors
    n_vectors = x.shape[1]
    cluster_labels = np.zeros([n_vectors], np.int32)

    # Means initialization
    if init_means is None:
        ind = np.random.choice(n_vectors, k, replace=False)
        centroids = x[:, ind]
    else:
        centroids = init_means

    i_iter = 0
    while i_iter < max_iter:
        raise NotImplementedError("You have to implement this function.")
        # Ploting partial results
        if show:
            print('Iteration: {:d}'.format(i_iter))
            show_clusters(x, cluster_labels, centroids, title='Iteration: {:d}'.format(i_iter))

    if show:
        print('Done.')

    raise NotImplementedError("You have to implement this function.")
    cluster_labels, centroids, sq_dists = None, None, None
    return cluster_labels, centroids, sq_dists


def k_means_multiple_trials(x, k, n_trials, max_iter, show=False):
    """
    Performs several trials of the k-centroids clustering algorithm in order to
    avoid local minima. Result of the trial with the lowest "within-cluster
    sum of squares" is selected as the best one and returned.

    :param x:               feature vectors, np array (dim, number_of_vectors)
    :param k:               required number of clusters, scalar
    :param n_trials:        number of trials, scalars
    :param max_iter:        stopping criterion: max. number of iterations
    :param show:            (optional) boolean switch to turn on/off visualization of partial results

    :return cluster_labels: cluster index for each feature vector, np array (number_of_vectors, ),
                            array contains only values from 0 to k-1,
                            i.e. cluster_labels[i] is the index of a cluster which the vector x[:,i] belongs to.
    :return centroids:      cluster centroids, np array (dim, k), same type as x
                            i.e. centroids[:,i] is the center of the i-th cluster.
    :return sq_dists:       squared distances to the nearest centroid for each feature vector,
                            np array (number_of_vectors, )
    """
    raise NotImplementedError("You have to implement this function.")
    cluster_labels, centroids, sq_dists = None, None, None
    return cluster_labels, centroids, sq_dists


def random_sample(weights):
    """
    picks randomly a sample based on the sample weights.

    :param weights: array of sample weights, np array (n, )
    :return idx:    index of chosen sample, scalar

    Note: use np.random.uniform() for random number generation in open interval (0, 1)
    """
    raise NotImplementedError("You have to implement this function.")
    idx = None
    return idx


def k_meanspp(x, k):
    """
    performs k-means++ initialization for k-means clustering.

    :param x:           Feature vectors, np array (dim, number_of_vectors)
    :param k:           Required number of clusters, scalar

    :return centroids:  proposed centroids for k-means initialization, np array (dim, k)
    """
    raise NotImplementedError("You have to implement this function.")
    centroids = None
    return centroids


def quantize_colors(im, k):
    """
    Image color quantization using the k-means clustering. A subset of 1000 pixels
    is first clustered into k clusters based on their RGB color.
    Quantized image is constructed by replacing each pixel color by its cluster centroid.

    :param im:          image for quantization, np array (h, w, 3) (np.uint8)
    :param k:           required number of quantized colors, scalar
    :return im_q:       image with quantized colors, np array (h, w, 3) (uint8)
    
    note: make sure that the k-means is run on floating point inputs.
    """

    assert im.dtype == np.uint8, f'input should be uint8, got {im.dtype}'
    assert im_q.dtype == np.uint8, f'output should be uint8, your output is {im_q.dtype}'

    raise NotImplementedError("You have to implement this function.")
    im_q = None
    return im_q


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


def compute_measurements(images):
    """
    computes 2D features from image measurements

    :param images: array of images, np array (H, W, N_images) (np.uint8)
    :return x:     array of features, np array (2, N_images)
    """

    images = images.astype(np.float64)
    H, W, N = images.shape

    left = images[:, :(W//2), :]
    right = images[:, (W//2):, :]
    up = images[:(H//2), ...]
    down = images[(H//2):, ...]

    L = np.sum(left, axis=(0, 1))
    R = np.sum(right, axis=(0, 1))
    U = np.sum(up, axis=(0, 1))
    D = np.sum(down, axis=(0, 1))

    a = L - R
    b = U - D

    x = np.vstack((a, b))
    return x


def show_clusters(x, cluster_labels, centroids, title=None, figsize=(4, 4)):
    """
    Create plot of feature vectors with same colour for members of same cluster.

    :param x:               feature vectors, np array (dim, number_of_vectors) (float64/double),
                            where dim is arbitrary feature vector dimension
    :param cluster_labels:  cluster index for each feature vector, np array (number_of_vectors, ),
                            array contains only values from 1 to k,
                            i.e. cluster_labels[i] is the index of a cluster which the vector x[:,i] belongs to.
    :param centroids:       cluster centers, np array (dim, k) (float64/double),
                            i.e. centroids[:,i] is the center of the i-th cluster.
    :param title:           optional parameter to set title of the figure, str
    """

    cluster_labels = cluster_labels.flatten()
    clusters = np.unique(cluster_labels)
    markers = itertools.cycle(['*', 'o', '+', 'x', 'v', '^', '<', '>'])

    plt.figure(figsize=figsize)
    for i in clusters:
        cluster_x = x[:, cluster_labels == i]
        # print(cluster_x)
        plt.plot(cluster_x[0], cluster_x[1], next(markers))
    plt.axis('equal')

    centroids_length = centroids.shape[1]
    for i in range(centroids_length):
        plt.plot(centroids[0, i], centroids[1, i], 'm+', ms=10, mew=2)

    plt.axis('equal')
    plt.grid('on')
    if title is not None:
        plt.title(title)


def show_clustered_images(images, labels, title=None):
    """
    Shows results of clustering. Create montages of images according to estimated labels

    :param images:          input images, np array (h, w, n)
    :param labels:          labels of input images, np array (n, )
    :param title:           optional parameter to set title of the figure, str
    """
    assert (len(images.shape) == 3)

    labels = labels.flatten()
    unique_labels = np.unique(labels)
    n = len(unique_labels)

    def montage(images, colormap='gray'):
        h, w, count = np.shape(images)
        h_sq = int(np.ceil(np.sqrt(count)))
        w_sq = h_sq
        im_matrix = np.ones((h_sq * h, w_sq * w))

        image_id = 0
        for j in range(h_sq):
            for k in range(w_sq):
                if image_id >= count:
                    break
                slice_w = j * h
                slice_h = k * w
                im_matrix[slice_h:slice_h + w, slice_w:slice_w + h] = images[:, :, image_id]
                image_id += 1
        return im_matrix

    width = int(min(n, 5))
    height = int(n // width + (n % width > 0))
    fig, axes = plt.subplots(height, width, figsize=(width * 2, height * 2))
    axes = axes.flatten()
    for i in range(n):
        plt.sca(axes[i])
        imgs = images[:, :, labels == unique_labels[i]]
        mont = montage(imgs)
        plt.imshow(mont, cmap='gray')
        plt.axis('off')

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()


def show_mean_images(images, labels, letters=None, title=None):
    """
    show_mean_images(images, c)

    Compute mean image for a cluster and show it.

    :param images:          input images, np array (h, w, n)
    :param labels:          labels of input images, np array (n, )
    :param letters:         labels for mean images, string/array of chars
    """
    assert (len(images.shape) == 3)

    labels = labels.flatten()
    l = np.unique(labels)
    n = len(l)

    unique_labels = np.unique(labels).flatten()

    fig, axes = plt.subplots(2, 5, figsize=(5, 2))
    axes = axes.flatten()

    for i in range(n):
        plt.sca(axes[i])
        imgs = images[:, :, labels == unique_labels[i]]
        img_average = np.squeeze(np.average(imgs.astype(np.float64), axis=2))
        plt.imshow(img_average, cmap='gray')
        if letters is not None:
            plt.title(letters[i])
        plt.axis('off')

    if title is not None:
        fig.suptitle(title)
        
    plt.tight_layout()



def gen_kmeanspp_data(mu=None, sigma=None, n=None):
    """
    generates data with n_clusterss normally distributed clusters

    It generates 4 clusters with 80 points by default.

    :param mu:          mean of normal distribution, np array (dim, n_clusters)
    :param sigma:       std of normal distribution, scalar
    :param n:           number of output points for each distribution, scalar
    :return samples:    dim-dimensional samples with n samples per cluster, np array (dim, n_clusters * n)
    """

    sigma = 1. if sigma is None else sigma
    mu = np.array([[-5, 0], [5, 0], [0, -5], [0, 5]]) if mu is None else mu
    n = 80 if n is None else n

    samples = np.random.normal(np.tile(mu, (n, 1)).T, sigma)
    return samples


def interactive_kmeans():
    """
    interactive visualisation of kmeans
    :return:
    """
    try:
        from ipywidgets import interact, interactive, fixed

        np.random.seed(0)
        x = gen_kmeanspp_data()

        @interact(k=(2, 8), n_iter=(0, 50, 1), seed=(0, 50, 1))
        def plot_k_means(k=4, n_iter=0, seed=0):
            np.random.seed(seed)
            centroids = x[:, np.random.choice(range(x.shape[1]), k, replace=False)]
            if n_iter == 0:
                show_clusters(x, np.ones([1, x.shape[1]]), centroids, title='K-means init')
            else:
                cluster_labels, centroids, _ = k_means(x, k, n_iter, False, centroids)
                show_clusters(x, cluster_labels, centroids, title='K-means {:d}-iters'.format(n_iter))

    except ImportError:
        print('Optional feature. If you want to play with interactive visualisations, '
              'you have to have installed ipywidgets and notebook has to be marked as Trusted')


def interactive_initialization_comparison():
    try:
        from ipywidgets import interact, interactive, fixed

        seed = 0
        np.random.seed(seed)
        x = gen_kmeanspp_data()

        @interact(k=(2, 8), n_iter=(0, 50, 1), seed=(0, 10, 1))
        def plot_k_means_init_comparison(k=4, n_iter=0, seed=0):
            cluster_labels = np.ones([x.shape[1]])
            cluster_labels_pp = np.ones([x.shape[1]])

            np.random.seed(seed)
            centroids = x[:, np.random.choice(range(x.shape[1]), k, replace=False)]
            np.random.seed(seed)
            centroids_pp = k_meanspp(x, k)

            if n_iter != 0:
                cluster_labels, centroids, _ = k_means(x, k, n_iter, False, centroids)
                cluster_labels_pp, centroids_pp, _ = k_means(x, k, n_iter, False, centroids_pp)

            show_clusters(x, cluster_labels, centroids, title='K-means random init ({:d}-iters)'.format(n_iter))
            show_clusters(x, cluster_labels_pp, centroids_pp, title='K-means kmeans++ init ({:d}-iters)'.format(n_iter))


    except ImportError:
        print('Optional feature. If you want to play with interactive visualisations, '
              'you have to have installed ipywidgets and notebook has to be marked as Trusted')

################################################################################
#####                                                                      #####
#####             Below this line you may insert debugging code            #####
#####                                                                      #####
################################################################################

def main():
    # HERE IT IS POSSIBLE TO ADD YOUR TESTING OR DEBUGGING CODE
    pass

if __name__ == "__main__":
    main()
