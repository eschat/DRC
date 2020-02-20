import numpy as np
import util
import pandas as pd
import math
import os

# Set working directory
os.chdir('/Users/Evelien/Desktop/DRC')


def simulation_part1(training_data, unseen_data, image_width, image_height, patch_width, patch_height, batch_size,
                     num_epochs, c_range, num_folds, num_patch_seq, iteration):

    """
    Part 1 of simulation, domain classification

    Parameters
    ----------
    training_data: str
        Specification of the training data, domain T
    unseen_data: str
        Specification of the unseen data, domain U
    image_width: int
        Width of image
    image_height: int
        Height of image
    patch_width: int
        Width of patches
    patch_height: int
        Height of patches
    batch_size: int
        Size of the batches to cut the data set into
    num_epochs: int
        Number of epochs in training a neural network
    c_range: array
        Range of possible regularization parameters
    num_folds: int
        Number of folds to use in cross-validation
    num_patch_seq: list
        Contains the different numbers of patches
    iteration: int
        Current iteration

    Returns
    -------
    results_pAD: array
        Proxy A-distance of domain classification for each iteration

    Note that the domain classification probabilities are saved in .csv files

    """

    ''' Subjects '''

    # 20 subjects
    brainwebSub = np.arange(20)

    # Number of subjects set I and set J
    nI = 15
    nJ = 5

    # Selecting subjects for set I and set J, random sampling
    sub_I = np.random.choice(brainwebSub, size=nI, replace=False)
    sub_J = np.random.choice(np.setdiff1d(brainwebSub, sub_I), size=nJ, replace=False)

    ''' Load Data - Phantoms '''

    # Preallocate data
    phantom_I_ = []
    phantom_J_ = []

    # Append
    for i in range(nI):
        phantom_I_.append("./phantoms/subject" + str(sub_I[i] + 1).zfill(2) + "_256.raw")

    for j in range(nJ):
        phantom_J_.append("./phantoms/subject" + str(sub_J[j] + 1).zfill(2) + "_256.raw")

    # Labels (Brainweb)
    phantom_I = util.subject2image(fn=phantom_I_, width=image_width, height=image_height, segmentation=True)
    phantom_J = util.subject2image(fn=phantom_J_, width=image_width, height=image_height, segmentation=True)

    ''' Load Data - Set I '''

    # Load set I for both the training data and the unseen data

    # Preallocate data
    training_data_I_ = []
    unseen_data_I_ = []

    # Append
    for i in range(nI):
        training_data_I_.append(
            "./data/" + training_data + "/subject" + str(sub_I[i] + 1).zfill(2) + "_256_" + training_data + ".raw")
        unseen_data_I_.append(
            "./data/" + unseen_data + "/subject" + str(sub_I[i] + 1).zfill(2) + "_256_" + unseen_data + ".raw")

    # Load set I
    training_data_I = util.subject2image(fn=training_data_I_, width=image_width, height=image_height, normalization=True)
    unseen_data_I = util.subject2image(fn=unseen_data_I_, width=image_width, height=image_height, normalization=True)

    ''' Load Data - Set J '''

    # Load set J for both the training dat and the unseen data

    # Preallocate data
    training_data_J_ = []
    unseen_data_J_ = []

    # Append
    for j in range(nJ):
        training_data_J_.append(
            "./data/" + training_data + "/subject" + str(sub_J[j] + 1).zfill(2) + "_256_" + training_data + ".raw")
        unseen_data_J_.append(
            "./data/" + unseen_data + "/subject" + str(sub_J[j] + 1).zfill(2) + "_256_" + unseen_data + ".raw")

    # Load set J
    training_data_J = util.subject2image(fn=training_data_J_, width=image_width, height=image_height, normalization=True)
    unseen_data_J = util.subject2image(fn=unseen_data_J_, width=image_width, height=image_height, normalization=True)

    ''' Pre-processing: strip skull '''

    # Create brain mask to strip the skull in simulated images, do this for set I and set J

    # Training data
    training_data_I = util.strip_skull(training_data_I, phantom_I == 0)
    training_data_J = util.strip_skull(training_data_J, phantom_J == 0)

    # Unseen data
    unseen_data_I = util.strip_skull(unseen_data_I, phantom_I == 0)
    unseen_data_J = util.strip_skull(unseen_data_J, phantom_J == 0)

    ''' Patch Extraction '''

    # Max amount of patches that can be extracted
    max_patch = (image_height - patch_height + 1) * (image_width - patch_width + 1)

    # Number of patches to be randomly sampled
    # Note that this is the last value in sequence of number of patches
    num_patch = num_patch_seq[-1]

    # Index of middle pixel in patch
    idx_mid = math.ceil((patch_height*patch_width)/2)

    # Extract patches
    # Training data, domain T
    training_I_patch = util.extract_patches(training_data_I, phantom_I, nI, max_patch, patch_width, patch_height,
                                            num_patch, idx_mid)
    training_J_patch = util.extract_patches(training_data_J, phantom_J, nJ, max_patch, patch_width, patch_height,
                                            num_patch, idx_mid)

    # Unseen data, domain U
    unseen_I_patch = util.extract_patches(unseen_data_I, phantom_I, nI, max_patch, patch_width, patch_height,
                                          num_patch, idx_mid)
    unseen_J_patch = util.extract_patches(unseen_data_J, phantom_J, nJ, max_patch, patch_width, patch_height,
                                          num_patch, idx_mid)

    ''' Simulation: Classification '''

    # Preallocate, will be used to save proxy A-distance results
    df_pAD = np.zeros([len(num_patch_seq), 1])

    for n in range(len(num_patch_seq)):

        # Patch selection
        # Training data
        training_set = util.patch_select(num_patch_seq, n, nI, nJ, patch_width, patch_height, training_I_patch[0],
                                         training_J_patch[0])
        training_I_patch_set = training_set[0]
        training_J_patch_set = training_set[1]

        # Unseen data
        unseen_set = util.patch_select(num_patch_seq, n, nI, nJ, patch_width, patch_height, unseen_I_patch[0],
                                       unseen_J_patch[0])
        unseen_I_patch_set = unseen_set[0]
        unseen_J_patch_set = unseen_set[1]

        ''' Domain Classification '''
        # Build a domain classifier on both the training and unseen data, and test the domain classifier on both the
        # training and unseen data

        # Domain labels: training data/domain T (0) and unseen data/domain U (1)
        # Domain T labels
        labels_domainT_build = np.zeros((num_patch_seq[n] * nI), dtype='int64')
        labels_domainT_test = np.zeros((num_patch_seq[n] * nJ), dtype='int64')

        # Domain U labels
        labels_domainU_build = np.ones((num_patch_seq[n] * nI), dtype='int64')
        labels_domainU_test = np.ones((num_patch_seq[n] * nJ), dtype='int64')

        # Combine labels
        labels_domain_build = np.concatenate((labels_domainT_build, labels_domainU_build))
        labels_domain_test = np.concatenate((labels_domainT_test, labels_domainU_test))

        # Combine data sets
        # Data set for building the domain classifier, combine training data and unseen data
        X_domain_build = np.concatenate((training_I_patch_set, unseen_I_patch_set), axis=0)

        # Data set for testing the domain classifier, combine training data and unseen data
        X_domain_test = np.concatenate((training_J_patch_set, unseen_J_patch_set), axis=0)

        # Domain Classification
        domain_classifier = util.classify(X_build=X_domain_build, X_test=X_domain_test, y_build=labels_domain_build,
                                          y_test=labels_domain_test, model='lr', patch_width=patch_width,
                                          patch_height=patch_height, num_folds=num_folds, c_range=c_range,
                                          batch_size=batch_size, num_epochs=num_epochs)

        # Obtain domain classification error
        domain_classification_error = domain_classifier[0]

        # Proxy A-distance
        pAD = 2 * (1 - (2 * domain_classification_error))
        df_pAD[n, :] = pAD

        # Obtain classification probabilities from domain classifier
        prob = pd.DataFrame(domain_classifier[1])
        prob_training = prob[0]  # probabilities training data
        prob_unseen = prob[1]  # probabilities unseen data

        # Save probabilities in .csv file, these will be used to obtain the DRC
        df_prob = pd.concat([prob_training, prob_unseen], axis=1)
        df_prob.to_csv("./results/results_probabilities/" + training_data + "_" + unseen_data + "/" + training_data +
                       "_" + unseen_data + "_nP" + str(num_patch_seq[n]) + "_prob_test" + str([iteration]) + ".csv",
                       index=False, header=False)

    # Reshape
    results_pAD = df_pAD.reshape(1, len(num_patch_seq))

    # Print proxy A-distance
    print('proxy A-distance', results_pAD)

    return results_pAD


''' Parameters '''

# Specify training data (scanner), domain T
training_data = "sc1"

# Specify unseen data (scanner), domain U
unseen_data = "sc2"
# unseen_data = "sc3"
# unseen_data = "sc4"
# unseen_data = "sc5"
# unseen_data = "sc6"
# unseen_data = "sc7"

# Image width
image_width = 256

# Image height
image_height = 256

# Patch width
patch_width = 15

# Patch height
patch_height = 15

# Batch size
batch_size = 128

# Number of epochs
num_epochs = 32

# Regularization
c_range = np.logspace(-5, 2, 15)

# Number of folds for cross-validation performance
num_folds = 2

# Number of iterations
num_repetition = 2

# Sequence number of patches
num_patch_seq = [100, 500, 1000, 2000, 3000, 4000, 5000]

''' Simulation '''

# Preallocate output data
df_domain_pAD = np.zeros([num_repetition, len(num_patch_seq)])

# Set seed, to ensure reproducibility, based on chosen data sets
# np.random.seed(2)     # training: sc1, unseen: sc1
np.random.seed(38)    # training: sc1, unseen: sc2
# np.random.seed(378)   # training: sc1, unseen: sc3
# np.random.seed(124)   # training: sc1, unseen: sc4
# np.random.seed(96)    # training: sc1, unseen: sc5
# np.random.seed(77)    # training: sc1, unseen: sc6
# np.random.seed(291)   # training: sc1, unseen: sc7

# Iterate, conduct simulation 50 times
for i in range(num_repetition):
    print("Iteration number", [i])
    sim_out = simulation_part1(training_data=training_data, unseen_data=unseen_data, image_width=image_width,
                               image_height=image_height, patch_width=patch_width, patch_height=patch_height,
                               batch_size=batch_size, num_epochs=num_epochs, c_range=c_range, num_folds=num_folds,
                               num_patch_seq=num_patch_seq, iteration=i)

    df_domain_pAD[i, :] = sim_out

# Save data as .csv file
df_domain_pAD = pd.DataFrame(df_domain_pAD)
df_domain_pAD.to_csv("./results/results_domain_classification/pAD/" + training_data + "_" + unseen_data +
                     "pAD.csv", index=False, header=False)
