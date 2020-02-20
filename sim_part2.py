import numpy as np
import scipy.stats as st
import util
import pandas as pd
import os

# Set working directory
os.chdir('/Users/Evelien/Desktop/DRC')


def simulation_part2(training_data, unseen_data, number_patch, iteration, beta1, beta2):

    """
    Part 2 of simulation, using probabilities obtained in simulation part 1 to obtain the Data Representativeness
    Criterion

    Parameters
    ----------
    training_data: str
        Specification of the training data, domain T
    unseen_data: str
        Specification of the unseen data, domain U
    number_patch:
        Number of patches
    iteration: int
        Current iteration
    beta1: int
        Alpha shape parameter for benchmark prior 1 distribution
    beta2: int
        Beta shape parameter for benchmark prior 1 distribution

    Returns
    -------
    kl_i: array
        Informative KL-divergence for each iteration
    kl_u: array
        Uninformative KL-divergence for each iteration
    drc: array
        Data Representativeness Criterion for each iteration

    Note that for the last two conditions (sc1_sc6 and sc1_sc7) the DRC could not be obtained, due to improper Beta
    distributions based on the probabilities

    """

    ''' Load data '''

    # Probabilities
    prob = pd.read_csv("results/results_domain_classification/results_probabilities/" + training_data + "_" +
                       unseen_data + "/" + training_data + "_" + unseen_data + "_nP" + str(number_patch) + "_prob" +
                       str([iteration]) + ".csv", header=None)

    # Specify probabilities of training data (domain T) and unseen data (domain U)
    prob_training = prob[0]
    prob_unseen = prob[1]

    ''' Obtaining beta distribution from the domain (scanner) probabilities '''

    # Generate linearly spaced vector with 10 million values
    bt = np.linspace(0, 1, 10000000)

    # Combine probabilities of training data and unseen data in one distribution
    comb_prob = np.concatenate((prob_training, prob_unseen), axis=0)

    # Obtain shape parameters
    shape_param = util.shape_beta(comb_prob)

    # Obtain beta distribution based on shape parameters
    beta_dist = st.beta.pdf(bt, shape_param[0], shape_param[1])

    ''' Benchmark prior distributions '''

    # Benchmark prior 1 - specified distribution
    bm_prior1 = st.beta.pdf(bt, beta1, beta2)

    # Benchmark prior 2 - uniform distribution
    bm_prior2 = st.beta.pdf(bt, 1, 1)

    # eps
    # Values below this threshold are replaced by this threshold for numerical stability
    eps = 10 ** (-20)
    beta_dist[beta_dist < eps] = eps
    bm_prior1[bm_prior1 < eps] = eps
    bm_prior2[bm_prior2 < eps] = eps

    ''' KL-divergences and Data Representativeness Criterion '''

    # KL-divergences
    kl_informative = st.entropy(beta_dist, bm_prior1)    # KL divergence specified benchmark prior
    kl_uninformative = st.entropy(beta_dist, bm_prior2)  # KL divergence uniform benchmark prior

    # Data Representativeness Criterion (DRC)
    drc = kl_informative / kl_uninformative
    print(drc)

    return kl_informative, kl_uninformative, drc


''' Parameters '''

# Specify training data (scanner), domain T
training_data = "sc1"

# Specify unseen data (scanner), domain U
unseen_data = "sc2"
# unseen_data = "sc3"
# unseen_data = "sc4"
# unseen_data = "sc5"

# Sequence number of patches
num_patch_seq = [100, 500, 1000, 2000, 3000, 4000, 5000]

# Number of iterations
num_repetition = 50

# Alpha shape parameter for benchmark prior 1
beta1 = 200

# Beta shape parameter for benchmark prior 1
beta2 = 200

# Iterate
for n in num_patch_seq:
    # Preallocate data
    df_kl_informative = np.zeros(num_repetition)
    df_kl_uninformative = np.zeros(num_repetition)
    df_drc = np.zeros(num_repetition)

    # Iterate
    for i in range(num_repetition):
        print([i])
        kl_informative, kl_uninformative, drc = simulation_part2(training_data=training_data, unseen_data=unseen_data,
                                                                 number_patch=n, iteration=i, beta1=beta1, beta2=beta2)

        df_kl_informative[i] = kl_informative
        df_kl_uninformative[i] = kl_uninformative
        df_drc[i] = drc

    # Data frames
    kl_informative_ = pd.DataFrame(df_kl_informative)
    kl_uninformative_ = pd.DataFrame(df_kl_uninformative)
    drc_ = pd.DataFrame(df_drc)

    # Concatenate
    df = pd.concat([kl_informative_, kl_uninformative_, drc_], axis=1)

    # Save data frame
    df.to_csv("./results/results_domain_classification/DRC/" + training_data + "_" + unseen_data + "/" + training_data +
              "_" + unseen_data + "_beta" + str(beta1) + "_" + str([n]) + ".csv", index=False, header=False)
