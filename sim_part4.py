import numpy as np
import util
import math
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
import sklearn.metrics as met

# Set working directory
os.chdir('/Users/Evelien/Desktop/DRC')

# Part 4 of simulation, tissue classification
# Tissue classifier built on training data and applied to unseen data
# Only 1 iteration


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

# Number of patches
num_patch = 7000

''' Simulation '''

# Set seed
np.random.seed(77)

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

# Load set I for the training data and the unseen data

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
training_data_I = util.subject2image(fn=training_data_I_, width=image_width, height=image_height,
                                     normalization=True)
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

# Index of middle pixel in patch, based on patches of 15x15
idx_mid = math.ceil((patch_height * patch_width) / 2)

# Extract patches
# Training data, domain T - 7,000 random patches per scan used for building the classifier
training_I_patch = util.extract_patches(training_data_I, phantom_I, nI, max_patch, patch_width, patch_height,
                                        num_patch, idx_mid)

# Unseen data, domain U - 7,000 random patches per scan used for testing the classifier.
# Note that this is only necessary for the util.classify function, but is not crucial for the results, as the model
# built on the training data will still be applied to 1 scan of the unseen data belonging to set J
unseen_I_patch = util.extract_patches(unseen_data_I, phantom_I, nI, max_patch, patch_width, patch_height, num_patch,
                                      idx_mid)

''' Tissue Classification '''

# Build a tissue classifier on training data, apply model to 1 scan of the unseen data and visualise predicted
# tissue classes in an image

# Obtain tissue labels: CSF(1), GM(2), WM(3)
y_training_I = util.tissue_label(training_I_patch[1], idx_mid)
y_unseen_I = util.tissue_label(unseen_I_patch[1], idx_mid)

# Recode tissue labels to [0, 1, 2] (CSF(0), GM(1), WM(2))
y_training_I = y_training_I - 1
y_unseen_I = y_unseen_I - 1

# Specify data sets
X_training_build = training_I_patch[0]
y_training_build = y_training_I

X_unseen_test = unseen_I_patch[0]
y_unseen_test = y_unseen_I

# Build classifier
tissue_classifier = util.classify(X_training_build, X_unseen_test, y_train=y_training_build.ravel(),
                                  y_test=y_unseen_test.ravel(), model='cnn', patch_width=patch_width,
                                  patch_height=patch_height, num_folds=num_folds, c_range=c_range,
                                  batch_size=batch_size, num_epochs=num_epochs)

# Save model
model = tissue_classifier[2]

# Next, the model will be applied to 1 unseen data scan from set J
# Obtain image
unseen_image = unseen_data_J[4]

# Extract all possible patches from the specified scan from the unseen data (unseen_image)
unseen_image_patch = image.extract_patches_2d(unseen_image, (patch_width, patch_height))

# Reshape
im_patch_shape = np.shape(unseen_image_patch)
unseen_image_patch = unseen_image_patch.reshape(im_patch_shape[0], im_patch_shape[1], im_patch_shape[2], 1)

# Obtain tissue class predictions based on the tissue classifier built on training data
unseen_predictions = model.predict_classes(unseen_image_patch)

# Recode tissue classes back to: CSF(1), GM(2), WM(3)
unseen_predictions = unseen_predictions + 1

# Reshape
unseen_predictions = unseen_predictions.reshape((242, 242))
unseen_predictions = np.array(unseen_predictions, dtype='int')

# Phantom
ph = phantom_J[4, 7:-7, 7:-7]
ph = np.array(ph, dtype='int')

# MRI scan
scan = unseen_data_J[4, 7:-7, 7:-7]

# Recode, brain mask
for s in range(np.shape(unseen_predictions)[0]):
    # Map pixels to 0 (background)
    unseen_predictions[s][ph[s] == 0] = 0

# Select pixels that are not background
ph_ = ph[ph != 0]
unseen_predictions_ = unseen_predictions[unseen_predictions != 0]

# Compare predicted tissue classes with ground truth (tissue classes in phantom)
print('tissue classification error', (1 - met.accuracy_score(ph_, unseen_predictions_)))

# Visualise predicted tissue classes of unseen data scan
# Specify Figure
fig = plt.figure(figsize=(15, 15))

# Visualise tissue class predictions, phantom, or MRI scan
plt.imshow(unseen_predictions[23:-23, 27:-27], vmin=0, vmax=3)
# plt.imshow(ph[23:-23, 27:-27], vmin=0, vmax=3)
# plt.imshow(scan[23:-23, 27:-27], cmap='gray', vmin=0, vmax=1)

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_visible(False)
cur_axes.axes.get_yaxis().set_visible(False)

# Save figure
# Tissue class predictions
fig.savefig('results/figures_paper/tissue_pred/' + training_data + '_' + unseen_data + 'test.png',
            bbox_inches='tight', pad_inches=0.1, dpi=1000)

# Phantom
# fig.savefig('results/results_tissue_classification/scans/phantom.png', bbox_inches='tight', pad_inches=0.1, dpi=1000)

# MRI Scan
# fig.savefig('results/results_tissue_classification/scans/scan.png', bbox_inches='tight', pad_inches=0.1, dpi=1000)
