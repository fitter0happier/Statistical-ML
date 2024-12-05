import numpy as np
import matplotlib.pyplot as plt


def adaboost(X, y, num_steps):
    """
    Trains an AdaBoost classifier

    :param X:                   training data containing feature points in columns, np array (d, n)
                                    d - number of weak classifiers
                                    n - number of data
    :param y:                   vector with labels (-1, 1) for feature points in X, np array (n, )
    :param num_steps:           maximum number of iterations

    :return strong_classifier:  dict with fields:
        - strong_classifier['wc'] - weak classifiers (see docstring of find_best_weak), np array (n_wc, )
        - strong_classifier['alpha'] - weak classifier coefficients, np array (n_wc, )
    :return wc_errors:          error of the best weak classifier in each iteration, np array (n_wc, )
    :return upper_bound:        upper bound on the training error in each iteration, np array (n_wc, )
    """

    alphas = np.zeros(num_steps)
    wc = np.empty(num_steps, dtype=object)

    strong_classifier = dict()
    strong_classifier['alpha'] = alphas
    strong_classifier['wc'] = wc

    upper_bound = np.ones(num_steps)
    wc_errors = np.zeros(num_steps)

    D = np.zeros_like(y, dtype=float)
    indices_neg = np.where(y == -1)[0]
    indices_pos = np.where(y == 1)[0]
    D[indices_neg] = 0.5 / len(indices_neg)
    D[indices_pos] = 0.5 / len(indices_pos)

    for t in range(num_steps):
        h_t, eps_t = find_best_weak(X, y, D)
        alpha_t = np.log((1 - eps_t) / eps_t) / 2
        Z_t = 2 * np.sqrt(eps_t * (1 - eps_t))

        wc[t] = h_t
        alphas[t] = alpha_t
        wc_errors[t] = eps_t
        upper_bound[t] = (upper_bound[t - 1] if t >= 0 else 1) * Z_t

        if (eps_t >= 0.5):
            break

        idx = h_t['idx']
        parity = h_t['parity']
        theta = h_t['theta']
        preds = np.where((parity * (X[idx] - theta)) > 0, 1, -1)

        D = (D / Z_t) * np.where(preds == y,
                                 np.sqrt(eps_t / (1 - eps_t)),
                                 np.sqrt((1 - eps_t) / eps_t))

    return strong_classifier, wc_errors, upper_bound


def adaboost_classify(strong_classifier, X):
    """ Classifies data X with a strong classifier

    :param strong_classifier:   classifier returned by adaboost (see docstring of adaboost)
    :param X:                   testing data containing feature points in columns, np array (d, n)
                                    d - number of weak classifiers
                                    n - number of data
    :return classif:            classification labels (values -1, 1), np array (n, )
    """
    classifs = strong_classifier['wc']
    alphas = strong_classifier['alpha']

    classif = np.zeros(X.shape[1])

    for i, x in enumerate(X.T):
        w_sum = np.sum([alpha * np.sign(classifier['parity'] * (x[classifier['idx']] - classifier['theta']))
                        for classifier, alpha in zip(classifs, alphas)])
        classif[i] = np.sign(w_sum)

    return classif


def compute_error(strong_classifier, X, y):
    """
    Computes the error on data X for all lengths of the given strong classifier

    :param strong_classifier:   classifier returned by adaboost - with T weak classifiers (see docstring of adaboost)
    :param X:                   testing data containing feature points in columns, np array (d, n)
                                    d - number of weak classifiers
                                    n - number of data
    :param y:                   testing labels (-1 or 1), np array (n, )
    :return errors:             errors of the strong classifier for all lengths from 1 to T, np array (T, )
    """

    T = len(strong_classifier['wc'])
    errors = np.zeros(T)

    for t in range(1, T + 1):
        classifs_up_to_t = strong_classifier['wc'][:t]
        alphas_up_to_t = strong_classifier['alpha'][:t]

        preds = np.zeros(X.shape[1])
        for i, x in enumerate(X.T):
            w_sum = np.sum([alpha * np.sign(classifier['parity'] * (x[classifier['idx']] - classifier['theta']))
                           for classifier, alpha in zip(classifs_up_to_t, alphas_up_to_t)])
            preds[i] = np.sign(w_sum)

        errors[t - 1] = np.sum(np.where(preds != y, 1, 0)) / y.shape[0]

    return errors


##########################################################################
#####
# Below this line are already prepared methods
#####
##########################################################################


def find_best_weak(X, y, D):
    """Finds best weak classifier

    Searches over all weak classifiers and their parametrisation
    (threshold and parity) for the weak classifier with lowest
    weighted classification error.

    The weak classifier realises following classification function:
        sign(parity * (x - theta))

    :param X:           training data containing feature points in columns, np array (d, n)
                            d - number of weak classifiers
                            n - number of data
    :param y:           vector with labels (-1, 1) for feature points in X, np array (n, )
    :param D:           training data weights, np array (n, )

    :return wc:         dict representing weak classifier with following fields:
        - wc['idx'] - index of the selected weak classifier, scalar
        - wc['theta'] - the classification threshold, scalar
        - wc['parity'] - the classification parity, scalar
    :return wc_error:   the weighted error of the selected weak classifier
    """
    assert X.ndim == 2
    assert y.ndim == 1
    assert y.size == X.shape[1]
    assert D.ndim == 1
    assert D.size == X.shape[1]

    N_wc, N = X.shape
    best_err = np.inf
    wc = {}

    for i in range(N_wc):
        weak_X = X[i, :]  # weak classifier evaluated on all data

        thresholds = np.unique(weak_X)
        assert thresholds.ndim == 1

        if thresholds.size > 1:
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2.
        else:
            thresholds = np.array([+1, -1] + thresholds[0])
        assert thresholds.ndim == 1

        K = thresholds.size

        classif = np.sign(np.reshape(weak_X, (N, 1)) -
                          np.reshape(thresholds, (1, K)))
        assert classif.ndim == 2
        assert classif.shape[0] == N
        assert classif.shape[1] == K

        # Broadcast
        column_D = np.reshape(D, (N, 1))
        column_y = np.reshape(y, (N, 1))
        err_pos = np.sum(column_D * (classif != column_y), axis=0)
        err_neg = np.sum(column_D * (-classif != column_y), axis=0)

        assert err_pos.ndim == 1
        assert err_pos.shape[0] == K
        assert err_neg.ndim == 1
        assert err_neg.shape[0] == K

        min_pos_idx = np.argmin(err_pos)
        min_pos_err = err_pos[min_pos_idx]

        min_neg_idx = np.argmin(err_neg)
        min_neg_err = err_neg[min_neg_idx]

        if min_pos_err < min_neg_err:
            err = min_pos_err
            parity = 1
            theta = thresholds[min_pos_idx]
        else:
            err = min_neg_err
            parity = -1
            theta = thresholds[min_neg_idx]

        if err < best_err:
            wc['idx'] = i
            wc['theta'] = theta
            wc['parity'] = parity
            best_err = err
    return wc, best_err


def show_classification(test_images, labels):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     np array (h, w, n)
    :param labels:          labels for input images np array (n,)
    """

    def montage(images, colormap='gray'):
        """
        Show images in grid.

        :param images:      np array (h, w, n)
        :param colormap:    numpy colormap
        """
        h, w, count = np.shape(images)
        h_sq = int(np.ceil(np.sqrt(count)))
        w_sq = h_sq
        im_matrix = np.zeros((h_sq * h, w_sq * w))

        image_id = 0
        for j in range(h_sq):
            for k in range(w_sq):
                if image_id >= count:
                    break
                slice_w = j * h
                slice_h = k * w
                im_matrix[slice_h:slice_h +
                          w, slice_w:slice_w +
                          h] = images[:, :, image_id]
                image_id += 1
        plt.imshow(im_matrix, cmap=colormap)
        plt.axis('off')
        return im_matrix

    imgs = test_images[..., labels == 1]
    subfig = plt.subplot(1, 2, 1)
    montage(imgs)
    plt.title('selected')

    imgs = test_images[..., labels == -1]
    subfig = plt.subplot(1, 2, 2)
    montage(imgs)
    plt.title('others')


def show_classifiers_part(class_images, classifier, max_alpha=None):
    """
    :param class_images:  images of a selected number, np array (h, w, n)
    :param classifier:    adaboost classifier
    """
    assert len(class_images.shape) == 3
    mean_image = np.mean(class_images, axis=2)
    mean_image = np.dstack((mean_image, mean_image, mean_image))
    vis = np.reshape(mean_image, (-1, 3))
    if max_alpha is None:
        max_alpha = np.amax(classifier['alpha'])

    for i, wc in enumerate(classifier['wc']):
        c = classifier['alpha'][i] / float(max_alpha)
        if wc['parity'] == 1:
            color = (c, 0, 0)
        else:
            color = (0, c, 0)
        vis[wc['idx'], :] = color

    vis = np.reshape(vis, mean_image.shape)
    plt.imshow(vis)
    plt.axis('off')


def show_classifiers(class_images, classifier):
    """
    :param class_images:  images of a selected number, np array (h, w, n)
    :param classifier:    adaboost classifier
    """
    # how many times each pixel is selected for a weak classifier:
    M, N = class_images.shape[:2]
    indices = [el['idx'] for el in classifier['wc']]
    counts, bins = np.histogram(indices, np.arange(-.5, M * N, 1))
    mx_count = np.max(counts)

    # create rounds of weak classifiers, at each round and for a given pixel,
    # there is at most 1 occurence:

    # keep max_alpha consistent between images:
    max_alpha = np.amax(classifier['alpha'])

    fig = plt.figure(figsize=(4 * mx_count, 4), dpi=100)
    done = [0 for k in range(len(indices))]
    shown = 0
    for k in range(mx_count):
        indices = np.zeros((M * N,))
        clf = {'wc': [], 'alpha': []}
        for i, (wc, alpha) in enumerate(
                zip(classifier['wc'], classifier['alpha'])):
            idx = wc['idx']
            if done[i] or indices[idx]:
                continue
            done[i] = 1
            indices[idx] = 1
            clf['wc'].append(wc)
            clf['alpha'].append(alpha)
            shown += 1
        subfig = plt.subplot(1, mx_count, k + 1)
        show_classifiers_part(class_images, clf, max_alpha=max_alpha)
    assert (shown == len(classifier['wc']))  # test if all shown


##########################################################################
#####
# Below this line you may insert debugging code
#####
##########################################################################

def main():
    # HERE IT IS POSSIBLE TO ADD YOUR TESTING OR DEBUGGING CODE
    pass


if __name__ == "__main__":
    main()
