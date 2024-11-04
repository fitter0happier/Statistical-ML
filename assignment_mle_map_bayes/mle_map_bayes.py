import numpy as np
from scipy.stats import norm
import scipy.special as spec  # for gamma
# importing bayes doesn't work in BRUTE :(, please copy the functions into this file


# MLE
def ml_estim_normal(x):
    """
    Computes maximum likelihood estimate of mean and variance of a normal distribution.

    :param x:   measurements, numpy array (n, )
    :return:    mu - mean - python float
                var - variance - python float
    """
    
    N = x.shape[0]
    mu = np.sum(x) / N
    var = np.sum((x - mu)**2) / N

    return mu, var


def ml_estim_categorical(counts):
    """
    Computes maximum likelihood estimate of categorical distribution parameters.

    :param counts: measured bin counts, numpy array (n, )
    :return:       pk - parameters of the categorical distribution, numpy array (n, )
    """
    
    N = np.sum(counts)
    pk = counts / N

    return pk

# MAP
def map_estim_normal(x, mu0, nu, alpha, beta):
    """
    Maximum a posteriori parameter estimation of normal distribution with normal inverse gamma prior.

    :param x:      measurements, numpy array (n, )
    :param mu0:    NIG parameter - python float
    :param nu:     NIG parameter - python float
    :param alpha:  NIG parameter - python float
    :param beta:   NIG parameter - python float

    :return:       mu - estimated mean - python float,
                   var - estimated variance - python float
    """
    
    N = x.shape[0]
    mu = (nu*mu0 + np.sum(x)) / (N + nu)
    var = (2*beta + nu*((mu0 - mu)**2) + np.sum((x - mu)**2)) / (N + 3 + 2*alpha)

    return mu, var


def map_estim_categorical(counts, alpha):
    """
    Maximum a posteriori parameter estimation of categorical distribution with Dirichlet prior.

    :param counts:  measured bin counts, numpy array (n, )
    :param alpha:   Dirichlet distribution parameters, numpy array (n, )

    :return:        pk - estimated categorical distribution parameters, numpy array (n, )
    """
    

    pk = (counts + alpha - 1) / np.sum(counts + alpha - 1)

    return pk

# BAYES
def bayes_posterior_params_normal(x, prior_mu0, prior_nu, prior_alpha, prior_beta):
    """
    Compute a posteriori normal inverse gamma parameters from data and NIG prior.

    :param x:            measurements, numpy array (n, )
    :param prior_mu0:    NIG parameter - python float
    :param prior_nu:     NIG parameter - python float
    :param prior_alpha:  NIG parameter - python float
    :param prior_beta:   NIG parameter - python float

    :return:             mu0:    a posteriori NIG parameter - python float
    :return:             nu:     a posteriori NIG parameter - python float
    :return:             alpha:  a posteriori NIG parameter - python float
    :return:             beta:   a posteriori NIG parameter - python float
    """

    N = x.shape[0]
    alpha = prior_alpha + N/2
    nu = prior_nu + N
    mu0 = (prior_nu*prior_mu0 + np.sum(x)) / (prior_nu + N)
    beta = (prior_beta 
        + np.sum(x**2)/2 
        + (prior_nu * prior_mu0**2)/2 
        - (prior_nu*prior_mu0 + np.sum(x))**2 / (2*(prior_nu + N)))
    
    return mu0, nu, alpha, beta

def bayes_posterior_params_categorical(counts, alphas):
    """
    Compute a posteriori Dirichlet parameters from data and Dirichlet prior.

    :param counts:   measured bin counts, numpy array (n, )
    :param alphas:   prior Dirichlet distribution parameters, numpy array (n, )

    :return:         posterior_alphas - estimated Dirichlet distribution parameters, numpy array (n, )
    """
    
    posterior_alphas = counts + alphas

    return posterior_alphas

def bayes_estim_pdf_normal(x_test, x,
                           mu0, nu, alpha, beta):
    """
    Compute pdf of predictive distribution for Bayesian estimate for normal distribution with normal inverse gamma prior.

    :param x_test:  values where the pdf should be evaluated, numpy array (m, )
    :param x:       'training' measurements, numpy array (n, )
    :param mu0:     prior NIG parameter - python float
    :param nu:      prior NIG parameter - python float
    :param alpha:   prior NIG parameter - python float
    :param beta:    prior NIG parameter - python float

    :return:        pdf - Bayesian estimate pdf evaluated at x_test, numpy array (m, )
    """

    (mu0_wave, 
    nu_wave, 
    alpha_wave, 
    beta_wave) = bayes_posterior_params_normal(x, mu0, nu, alpha, beta)

    alpha_hat = alpha_wave + 1/2
    nu_hat = nu_wave + 1
    beta_hat = (x_test**2 / 2 + beta_wave 
                + (nu_wave * mu0_wave**2) / 2
                - (nu_wave * mu0_wave + x_test)**2  / (2 * (nu_wave + 1)))

    gamma_alpha_wave = spec.gamma(alpha_wave)
    gamma_alpha_hat = spec.gamma(alpha_hat)
    sqrt2pi = np.sqrt(2*np.pi)

    pdf = ((np.sqrt(nu_wave) 
            * beta_wave**alpha_wave 
            * gamma_alpha_hat)
            / 
            (sqrt2pi 
            * np.sqrt(nu_hat) 
            * (beta_hat)**alpha_hat 
            * gamma_alpha_wave)) 
    
    return pdf

def bayes_estim_categorical(counts, alphas):
    """
    Compute parameters of Bayesian estimate for categorical distribution with Dirichlet prior.

    :param counts:  measured bin counts, numpy array (n, )
    :param alphas:  prior Dirichlet distribution parameters, numpy array (n, )

    :return:        pk - estimated categorical distribution parameters, numpy array (n, )
    """

    alphas_wave = bayes_posterior_params_categorical(counts, alphas)
    pk = alphas_wave / np.sum(alphas_wave)

    return pk

# Classification
def mle_Bayes_classif(x_test, x_train_A, x_train_C):
    """
    Classify images using Bayes classification using MLE of normal distributions and 0-1 loss.

    :param x_test:         test image features, numpy array (N, )
    :param x_train_A:      training image features A, numpy array (nA, )
    :param x_train_C:      training image features C, numpy array (nC, )

    :return:               q - classification strategy (see find_strategy_2normal)
    :return:               labels - classification of test_data, numpy array (N, ) (see bayes.classify_2normal)
    :return:               DA - parameters of the normal distribution of A
                            DA['Mean'] - python float
                            DA['Sigma'] - python float
                            DA['Prior'] - python float
    :return:               DC - parameters of the normal distribution of C
                            DC['Mean'] - python float
                            DC['Sigma'] - python float
                            DC['Prior'] - python float
    """
    DA = dict()
    DC = dict()

    N = x_train_A.shape[0] + x_train_C.shape[0]

    DA['Mean'], DA['Sigma'] = ml_estim_normal(x_train_A)
    DA['Sigma'] = np.sqrt(DA['Sigma'])
    DA['Prior'] = x_train_A.shape[0] / N

    DC['Mean'], DC['Sigma'] = ml_estim_normal(x_train_C)
    DC['Sigma'] = np.sqrt(DC['Sigma'])
    DC['Prior'] = x_train_C.shape[0] / N

    q = find_strategy_2normal(DA, DC)
    labels = classify_2normal(x_test, q)

    return q, labels, DA, DC


def map_Bayes_classif(x_test, x_train_A, x_train_C,
                      mu0_A, nu_A, alpha_A, beta_A,
                      mu0_C, nu_C, alpha_C, beta_C):
    """
    Classify images using Bayes classification using MAP estimate of normal distributions with NIG priors and 0-1 loss.

    :param x_test:         test image features, numpy array (N, )
    :param x_train_A:      training image features A, numpy array (nA, )
    :param x_train_C:      training image features C, numpy array (nC, )

    :param mu0_A:          prior NIG parameter for A - python float
    :param nu_A:           prior NIG parameter for A - python float
    :param alpha_A:        prior NIG parameter for A - python float
    :param beta_A:         prior NIG parameter for A - python float

    :param mu0_C:          prior NIG parameter for C - python float
    :param nu_C:           prior NIG parameter for C - python float
    :param alpha_C:        prior NIG parameter for C - python float
    :param beta_C:         prior NIG parameter for C - python float

    :return:               q - classification strategy (see find_strategy_2normal)
    :return:               labels - classification of test_imgs, numpy array (N, ) (see bayes.classify_2normal)
    :return:               DA - parameters of the normal distribution of A
                            DA['Mean'] - python float
                            DA['Sigma'] - python float
                            DA['Prior'] - python float
    :return:               DC - parameters of the normal distribution of C
                            DC['Mean'] - python float
                            DC['Sigma'] - python float
                            DC['Prior'] - python float

    """
    DA = dict()
    DC = dict()

    N = x_train_A.shape[0] + x_train_C.shape[0]

    DA['Mean'], DA['Sigma'] = map_estim_normal(x_train_A, 
                                               mu0_A, 
                                               nu_A, 
                                               alpha_A, 
                                               beta_A)
    DA['Sigma'] = np.sqrt(DA['Sigma'])
    DA['Prior'] = x_train_A.shape[0] / N

    DC['Mean'], DC['Sigma'] = map_estim_normal(x_train_C, 
                                               mu0_C, 
                                               nu_C, 
                                               alpha_C, 
                                               beta_C)
    DC['Sigma'] = np.sqrt(DC['Sigma'])
    DC['Prior'] = x_train_C.shape[0] / N

    q = find_strategy_2normal(DA, DC)
    labels = classify_2normal(x_test, q)

    return q, labels, DA, DC


def bayes_Bayes_classif(x_test, x_train_A, x_train_C,
                        mu0_A, nu_A, alpha_A, beta_A,
                        mu0_C, nu_C, alpha_C, beta_C):
    """
    Classify images using Bayes classification (0-1 loss) using predictive pdf estimated using Bayesian inference with NIG priors.

    :param x_test:         images features to be classified, numpy array (n, )
    :param x_train_A:      training image features A, numpy array (nA, )
    :param x_train_C:      training image features C, numpy array (nC, )

    :param mu0_A:          prior NIG parameter for A - python float
    :param nu_A:           prior NIG parameter for A - python float
    :param alpha_A:        prior NIG parameter for A - python float
    :param beta_A:         prior NIG parameter for A - python float

    :param mu0_C:          prior NIG parameter for C - python float
    :param nu_C:           prior NIG parameter for C - python float
    :param alpha_C:        prior NIG parameter for C - python float
    :param beta_C:         prior NIG parameter for C - python float

    :return:               labels - classification of x_test, numpy array (n, ) int32, values 0 or 1
    """

    N = x_train_A.shape[0] + x_train_C.shape[0]

    pA = x_train_A.shape[0] / N
    pC = x_train_C.shape[0] / N
    
    pdf_A = bayes_estim_pdf_normal(x_test, 
                                   x_train_A, 
                                   mu0_A, 
                                   nu_A, 
                                   alpha_A, 
                                   beta_A)
    
    pdf_C = bayes_estim_pdf_normal(x_test,
                                   x_train_C,
                                   mu0_C,
                                   nu_C,
                                   alpha_C,
                                   beta_C)
    
    labels = np.where(pA * pdf_A >= pC * pdf_C, 0, 1)

    return labels


################################################################################
#####                                                                      #####
#####                Put functions from previous labs here.                #####
#####            (Sorry, we know imports would be much better)             #####
#####                                                                      #####
################################################################################


def find_strategy_2normal(distribution_A, distribution_B):
    """
    q = find_strategy_2normal(distribution_A, distribution_B)

    Find optimal bayesian strategy for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior'] - python floats
    :param distribution_B:  the same as distribution_A

    :return q:              strategy dict
                               q['t1'], q['t2'] - decision thresholds - python floats
                               q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
                               If there is only one threshold, q['t1'] should be equal to q['t2'] and the middle decision should be 0
                               If there is no threshold, q['t1'] and q['t2'] should be -/+ infinity and all the decision values should be the same (0 preferred)
    """
    
    s_A = distribution_A['Sigma']
    m_A = distribution_A['Mean']
    p_A = distribution_A['Prior']
    s_B = distribution_B['Sigma']
    m_B = distribution_B['Mean']
    p_B = distribution_B['Prior']

    q = {}

    # extreme priors
    eps = 1e-10
    if p_A < eps:
        q['t1'] = -np.inf
        q['t2'] = np.inf
        q['decision'] = np.array([1, 1, 1], dtype=np.int32)

    elif p_B < eps:
        q['t1'] = -np.inf
        q['t2'] = np.inf
        q['decision'] = np.array([0, 0, 0], dtype=np.int32)

    else:
        a = - (1 / (2 * (s_A ** 2))) + (1 / (2 * (s_B ** 2)))
        b = (m_A / s_A ** 2) - (m_B / s_B ** 2)
        c = - (m_A ** 2 / (2 * (s_A ** 2))) + (m_B ** 2 / (2 * (s_B ** 2))) + np.log((s_B * p_A) / (s_A * p_B))

        if a == 0:
            # same sigmas -> not quadratic
            if b == 0:
                # same sigmas and same means -> not even linear
                q['t1'] = -np.inf
                q['t2'] = np.inf

                if (p_A >= p_B):
                    q['decision'] = np.array([0, 0, 0], dtype=np.int32)

                else:
                    q['decision'] = np.array([1, 1, 1], dtype=np.int32)
            else:
                # same sigmas, different means -> linear equation
                t_1 = t_2 = - c / b
                q['t1'] = t_1
                q['t2'] = t_2

                if (b > 0):
                    q['decision'] = np.array([1, 0, 0], dtype=np.int32)

                else:
                    q['decision'] = np.array([0, 0, 1], dtype=np.int32)

        else:
            # quadratic equation
            D = b ** 2 - 4 * a * c

            if D > 0:
                t_1 = (- b + np.sqrt(D)) / (2 * a)
                t_2 = (- b - np.sqrt(D)) / (2 * a)

                if (t_1 < t_2):
                    q['t1'] = t_1
                    q['t2'] = t_2

                else:
                    q['t1'] = t_2
                    q['t2'] = t_1

                if (a < 0):
                    q['decision'] = np.array([1, 0, 1], dtype=np.int32)

                else:
                    q['decision'] = np.array([0, 1, 0], dtype=np.int32)

            elif D == 0:
                t_1 = t_2 = - b / (2 * a)
                q['t1'] = t_1
                q['t2'] = t_2

                if (a < 0):
                    q['decision'] = np.array([1, 0, 1], dtype=np.int32)

                else:
                    q['decision'] = np.array([0, 0, 0], dtype=np.int32)

            elif D < 0:
                q['t1'] = -np.inf
                q['t2'] = np.inf

                if (a < 0):
                    q['decision'] = np.array([1, 1, 1], dtype=np.int32)

                else:
                    q['decision'] = np.array([0, 0, 0], dtype=np.int32)
    return q


def classify_2normal(measurements, q):
    """
    label = classify_2normal(measurements, q)

    Classify images using continuous measurements and strategy q.

    :param imgs:    test set measurements, np.array (n, )
    :param q:       strategy
                    q['t1'] q['t2'] - float decision thresholds
                    q['decision'] - (3, ) int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:        label - classification labels, (n, ) int32
    """
    
    labels = None

    if (q['t1'] == q['t2']):
        labels = np.where(measurements <= q['t1'], q['decision'][0], q['decision'][2])

    elif (q['t2'] == np.inf):
        labels = np.full(q['decision'][0], measurements.shape[0])

    else:
        labels = np.where(measurements <= q['t1'], q['decision'][0], np.where(measurements <= q['t2'], q['decision'][1], q['decision'][2]))

    return labels


def classification_error(predictions, labels):
    """
    error = classification_error(predictions, labels)

    :param predictions: (n, ) np.array of values 0 or 1 - predicted labels
    :param labels:      (n, ) np.array of values 0 or 1 - ground truth labels
    :return:            error - classification error ~ a fraction of predictions being incorrect
                        python float in range <0, 1>
    """
    
    error = np.average(np.where(predictions == labels, 0, 1))

    return error


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


def compute_measurement_lr_cont(imgs):
    """
    x = compute_measurement_lr_cont(imgs)

    Compute measurement on images, subtract sum of right half from sum of
    left half.

    :param imgs:    set of images, (h, w, n) numpy array
    :return x:      measurements, (n, ) numpy array
    """
    assert len(imgs.shape) == 3

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2),:], axis=0) - np.sum(sum_rows[int(width / 2):,:], axis=0)

    assert x.shape == (imgs.shape[2], )
    return x


def mle_likelihood_normal(x, mu, var):
    """
    Compute the likelihood of the data x given the model is a normal distribution with given mean and sigma

    :param x:       measurements, numpy array (n, )
    :param mu:      the normal distribution mean
    :param var:     the normal distribution variance
    :return:        L - likelihood of the data x
    """
    assert len(x.shape) == 1

    if var <= 0:
        L = 0
    else:
        L = np.prod(norm.pdf(x, mu, np.sqrt(var)))
    return L


def norm_inv_gamma_pdf(mu, var, mu0, nu, alpha, beta):
    # Wikipedia sometimes uses a symbol 'lambda' instead 'nu'

    assert alpha > 0
    assert nu > 0
    if beta <= 0 or var <= 0:
        return 0

    sigma = np.sqrt(var)

    p = np.sqrt(nu) / (sigma * np.sqrt(2 * np.pi)) * np.power(beta, alpha) / spec.gamma(alpha) * np.power(1/var, alpha + 1) * np.exp(-(2 * beta + nu * (mu0 - mu) * (mu0 - mu)) / (2 * var))

    return p


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
