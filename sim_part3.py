import numpy as np
import pandas as pd
import util
import math
import os

# Set working directory
os.chdir('/Users/Evelien/Desktop/DRC')


def simulation_part3(training_data, unseen_data, image_width, image_height, patch_width, patch_height, batch_size,
                     num_epochs, c_range, num_folds, num_patch, num_patch_unseen):

    """
    Part 3 of simulation, tissue classification
    Two classifiers were used: 1) training classifier (training + unseen): a convolution neural network (CNN) built
    on both training and unseen data and 2) unseen classifier (unseen): a CNN built only on unseen data

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
    num_patch: int
        Number of patches to be extracted for building and testing classifier
    num_patch_unseen: list
        Contains the different numbers of target patches that will be added to training set

    Returns
    -------
    tissue_error: array
        Tissue classification error for each iteration

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

    # Load set J for the unseen data

    # Preallocate data
    unseen_data_J_ = []

    # Append
    for j in range(nJ):
        unseen_data_J_.append(
            "./data/" + unseen_data + "/subject" + str(sub_J[j] + 1).zfill(2) + "_256_" + unseen_data + ".raw")

    # Load set J
    unseen_data_J = util.subject2image(fn=unseen_data_J_, width=image_width, height=image_height, normalization=True)

    ''' Pre-processing: strip skull '''

    # Create brain mask to strip the skull in simulated images, do this for set I and set J

    # Training data
    training_data_I = util.strip_skull(training_data_I, phantom_I == 0)

    # Unseen data
    unseen_data_I = util.strip_skull(unseen_data_I, phantom_I == 0)
    unseen_data_J = util.strip_skull(unseen_data_J, phantom_J == 0)

    ''' Patch Extraction '''

    # Max amount of patches that can be extracted
    max_patch = (image_height - patch_height + 1) * (image_width - patch_width + 1)

    # Index of middle pixel in patch
    idx_mid = math.ceil((patch_height * patch_width) / 2)

    # Extract patches
    # Training data, domain T - 7,000 random patches per scan used for building the classifier
    training_I_patch = util.extract_patches(training_data_I, phantom_I, nI, max_patch, patch_width, patch_height,
                                            num_patch, idx_mid)

    # Unseen data, domain U - 7,000 random patches per scan used for testing the classifier
    unseen_I_patch = util.extract_patches(unseen_data_I, phantom_I, nI, max_patch, patch_width, patch_height,
                                          num_patch, idx_mid)

    ''' Simulation: Classification '''

    # Preallocate
    df_training_unseen = np.zeros([len(num_patch_unseen), 1])
    df_unseen = np.zeros([len(num_patch_unseen), 1])

    for n in range(len(num_patch_unseen)):

        ''' Tissue Classification '''

        # Extract patches
        # Unseen data, domain U - 100-18,000 random patches per scan which will be added to the training data and
        # together used to build the classifier
        unseen_J_patch = util.extract_patches(unseen_data_J, phantom_J, nJ, max_patch, patch_width, patch_height,
                                              num_patch_unseen[n], idx_mid)

        # Obtain tissue labels: CSF(1), GM(2), WM(3)
        y_training_I = util.tissue_label(training_I_patch[1], idx_mid)
        y_unseen_I = util.tissue_label(unseen_I_patch[1], idx_mid)
        y_unseen_J = util.tissue_label(unseen_J_patch[1], idx_mid)

        # Recode tissue labels to [0, 1, 2] (CSF(0), GM(1), WM(2))
        y_training_I = y_training_I - 1
        y_unseen_I = y_unseen_I - 1
        y_unseen_J = y_unseen_J - 1

        # Classifier 1: training + unseen CNN, built on both patches from the training data and unseen data
        # Combine data sets
        X_tr_un_build = np.concatenate((training_I_patch[0], unseen_J_patch[0]), axis=0)
        y_tr_un_build = np.concatenate((y_training_I, y_unseen_J), axis=0)

        # Specify additional data sets
        X_tr_un_test = unseen_I_patch[0]
        y_tr_un_test = y_unseen_I

        # Build and test classifier 1
        tissue_classifier_tr_un = util.classify(X_build=X_tr_un_build, X_test=X_tr_un_test,
                                                y_build=y_tr_un_build.ravel(), y_test=y_tr_un_test.ravel(),
                                                model='cnn', patch_width=patch_width, patch_height=patch_height,
                                                num_folds=num_folds, c_range=c_range, batch_size=batch_size,
                                                num_epochs=num_epochs)

        # Classification error
        error_tr_un = tissue_classifier_tr_un[0]
        df_training_unseen[n, :] = error_tr_un

        # Classifier 2: unseen CNN, built on only unseen data
        # Specify data sets
        X_unseen_build = unseen_J_patch[0]
        y_unseen_build = y_unseen_J
        X_unseen_test = unseen_I_patch[0]
        y_unseen_test = y_unseen_I

        # Build and test classifier 2
        tissue_classifier_unseen = util.classify(X_build=X_unseen_build, X_test=X_unseen_test,
                                                 y_build=y_unseen_build.ravel(), y_test=y_unseen_test.ravel(),
                                                 model='cnn', patch_width=patch_width, patch_height=patch_height,
                                                 num_folds=num_folds, c_range=c_range, batch_size=batch_size,
                                                 num_epochs=num_epochs)

        # Classification error
        error_unseen = tissue_classifier_unseen[0]
        df_unseen[n, :] = error_unseen

    # Reshape
    tissue_error_tr_un = df_training_unseen.reshape(1, len(num_patch_unseen))
    tissue_error_unseen = df_unseen.reshape(1, len(num_patch_unseen))

    # Print tissue classification errors
    print('classification error training+unseen classifier', tissue_error_tr_un)
    print('classification error unseen classifier', tissue_error_unseen)

    return tissue_error_tr_un, tissue_error_unseen


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
num_repetition = 10

# Number of patches
num_patch = 7000

# Sequence number of target patches to be added to training set
num_patch_unseen = [100, 500, 1000, 2000, 5000, 10000, 12000, 15000, 18000]

''' Simulation '''

# Preallocate for tissue classification performance results
performance_tr_un = np.zeros([num_repetition, len(num_patch_unseen)])
performance_unseen = np.zeros([num_repetition, len(num_patch_unseen)])

# Set seed, to ensure reproducibility, based on chosen scanners
np.random.seed(38)    # training: sc1, unseen: sc2
# np.random.seed(378)   # training: sc1, unseen: sc3
# np.random.seed(124)   # training: sc1, unseen: sc4
# np.random.seed(96)    # training: sc1, unseen: sc5
# np.random.seed(77)    # training: sc1, unseen: sc6
# np.random.seed(291)   # training: sc1, unseen: sc7

# Iterate, conduct simulation 10 times
for i in range(num_repetition):
    print([i])
    sim_out = simulation_part3(training_data=training_data, unseen_data=unseen_data, image_width=image_width,
                               image_height=image_height, patch_width=patch_width, patch_height=patch_height,
                               batch_size=batch_size, num_epochs=num_epochs, c_range=c_range,
                               num_folds=num_folds, num_patch=num_patch, num_patch_unseen=num_patch_unseen)

    performance_tr_un[i, :] = sim_out[0]
    performance_unseen[i, :] = sim_out[1]

# Save data as .csv file
df_cl_tr_un = pd.DataFrame(performance_tr_un)
df_cl_tr_un.to_csv("./results/results_tissue_classification/" + training_data + "_" + unseen_data +
                   "_classification_tr_un.csv", index=False, header=False)

df_cl_unseen = pd.DataFrame(performance_unseen)
df_cl_unseen.to_csv("./results/results_tissue_classification/" + training_data + "_" + unseen_data +
                    "_classification_unseen.csv", index=False, header=False)
