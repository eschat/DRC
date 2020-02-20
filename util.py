import numpy as np
import nibabel as nib

from sklearn import svm
from sklearn.feature_extraction import image
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
import sklearn.svm as sksv

import scipy.ndimage as nd
from scipy.ndimage.interpolation import zoom

import keras.models as km
import keras.layers as kl
import keras.regularizers as kr
from keras.utils import to_categorical

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()

# R packages
ro.conversion.py2ri = numpy2ri
ro.numpy2ri.activate()
base = importr('base')
utils = importr('utils')
fitdistr = importr('fitdistrplus')
mass = importr('flexmix')
dist = ro.r['fitdist']
vector = ro.r['as.vector']
matrix = ro.r['as.matrix']
kl_d = ro.r['KLdiv']
cbind = ro.r['cbind']
infinite = ro.r['is.infinite']


def subject2image(fn, width, height, slice_ix=0, slice_dim=2, segmentation=False, classes=[0, 1, 2, 3], flip=False,
                  normalization=False):

    """
    https://github.com/wmkouw/mrai-net

    Load subject images

    Parameters
    ----------
    fn: list[str]
        Filenames of images
    width: int
        Width of image
    height: int
        Height of image
    slice_ix: int
        Slice index
    slice_dim: int
        Slice dimensions
    segmentation: bool
        Indicates whether it is a label matrix
    classes: list[int]
        List of numerical values of tissue classes
    flip: bool
        Indicated whether to flip image
    normalization: bool
        Indicated whether to normalize images

    Returns
    -------
    X:  array
        Loaded images (number of subjects, image height, image width

    """

    # Find number of subjects
    nS = len(fn)

    # Preallocate
    X = np.empty((nS, width, height))

    # Loop over subjects
    for s in range(nS):

        # Recognize file format
        file_fmt = fn[s][-3:]

        # Check for file format
        if file_fmt == 'raw':

            # Read binary and reshape to image
            im = nd.rotate(np.fromfile(fn[s], count=width * height, dtype='uint8').reshape((width, height)), 90)

            if segmentation:

                # Restrict classes of segmentations
                labels = np.unique(im)
                for lab in np.setdiff1d(labels, classes):
                    im[im == lab] = 0

                X[s, :, :] = im

            else:
                # Cubic spline interpolation for zoom
                # im = np.round(zoom(im[31:217,25:230].astype(np.float64), zd, order=3)).astype('int64')

                # Normalize pixels
                if normalization:
                    im[im < 0] = 0
                    im[im > 255] = 255
                    im = im / 255.

                # Place zoom into image array
                # I[s,22:227,42:213] = im
                X[s, :, :] = im

        elif file_fmt == 'nii':

            # Collect image and pixel dims
            im = nib.load(fn[s]).get_data()
            hd = nib.load(fn[s]).header
            zd = hd.get_zooms()

            if len(im.shape) == 4:
                im = im[:, :, :, 0]
                zd = zd[:-1]

            if segmentation:
                im = CMA23(im)

            # Interpolate to normalized pixel dimension
            im = zoom(im, zd, order=0)

            # Pad image
            pdl = np.ceil((np.array((256, 256, 256)) - im.shape) / 2.).astype('int64')
            pdh = np.floor((np.array((256, 256, 256)) - im.shape) / 2.).astype('int64')
            im = np.pad(im, ((pdl[0], pdh[0]), (pdl[1], pdh[1]), (pdl[2], pdh[2])), 'constant')

            # Normalize pixels
            if normalization:
                im[im < 0] = 0
                # im[im>hd.sizeof_hdr] = hd.sizeof_hdr
                im = im / float(1023)

            # Slice image
            if slice_dim == 2:
                X[s, :, :] = im[:, :, slice_ix].T
            elif slice_dim == 1:
                X[s, :, :] = im[:, slice_ix, :].T
            else:
                X[s, :, :] = im[slice_ix, :, :].T

            if flip:
                X[s, :, :] = np.flipud(X[s, :, :])

        else:
            print('File format unknown')

    return X


def strip_skull(X, M):

    """
    https://github.com/wmkouw/mrai-net

    Strip skull of image X based on mask M

    Parameters
    ----------
    X:  array
        MRI scans that include skulls of subjects
    M:  boolean array
        MRI scans that indicate the skulls

    Returns
    -------
    X:  array
        MRI scans with 0's at mask indices

    """

    # Check if X contains multiple subjects
    if len(X.shape) > 2:

        # Check whether data and mask have same number of subjects
        if not (X.shape[0] == M.shape[0]):
            raise ValueError('Shape mismatch.')

        # Strip skull for each subject
        for i in range(X.shape[0]):
            X[i][M[i]] = 0

    else:
        # Strip skull
        X[M] = 0

    return X


def extract_patches(X, P, num_subject, max_patch, patch_width, patch_height, num_patch, idx_mid):

    """
    Extract patches from image

    Parameters
    ----------
    X:  array
        MRI scans to extract patches from
    P:  array
        Phantoms of MRI scans to extract patches from
    num_subject: int
        Number of subjects
    max_patch: int
        Max amount of patches that can be extracted from one image, with dimensions patch_width x patch_height
    patch_width: int
        Width of patches to extract
    patch_height: int
        Height of patches to extract
    num_patch: int
        Number of patches to be randomly sampled
    idx_mid: int
        Index of middle of patch, with dimensions patch_width x patch_height

    Returns
    -------
    patch_scans: array
        Sample of patches from X
    patch_tissue: array
        Sample of patches from P

    """

    # Preallocate
    df_X = np.zeros(((num_subject * max_patch), patch_width, patch_height))
    df_P = np.zeros(((num_subject * max_patch), patch_width, patch_height))

    # Set c to 0 
    c = 0

    for i in range(X.shape[0]):
        # Extract max amount of patches from I
        df_X[c:c + max_patch] = (image.extract_patches_2d(X[i], (patch_width, patch_height)))

        # Extract max amount of patches from P
        df_P[c:c + max_patch] = (image.extract_patches_2d(P[i], (patch_width, patch_height)))

        # Tick up
        c = c + max_patch

    # Reshape arrays
    df_X = df_X.reshape((-1, patch_width * patch_height))
    df_X = df_X.reshape((num_subject, max_patch, patch_width * patch_height))
    
    df_P = df_P.reshape((-1, patch_width * patch_height))
    df_P = df_P.reshape((num_subject, max_patch, patch_width * patch_height))

    # Preallocate
    patch_scans = np.zeros((num_subject, num_patch, patch_width * patch_height))
    patch_tissue = np.zeros((num_subject, num_patch, patch_width * patch_height))

    for i in range(num_subject):

        # Mask brain, to include less background; filter out patches where middle pixel contains background
        df_X_mask = df_X[i, :, :][df_X[i, :, idx_mid] != 0]
        df_P_mask = df_P[i, :, :][df_P[i, :, idx_mid] != 0]

        # Randomly sample nP number of index numbers for patches, without replacement
        idx = np.random.choice(df_X_mask.shape[0], num_patch, replace=False)

        # Obtain patches of X and P based on randomly sampled index numbers
        patch_scans[i] = df_X_mask[idx, :]
        patch_tissue[i] = df_P_mask[idx, :]

    # Reshape
    patch_scans = patch_scans.reshape((-1, patch_width * patch_height))
    patch_tissue = patch_tissue.reshape((-1, patch_width * patch_height))

    return patch_scans, patch_tissue


def patch_select(num_patch_seq, n, nI, nJ, patch_width, patch_height, dataI, dataJ):

    """
    Select patches
    Select num_patch_seq[n] patches of each subject

    Parameters
    ----------
    num_patch_seq: list
        Contains the different numbers of patches
    n:  int
        n'th number in nP_seq
    nI: int
        Number of subjects set I
    nJ: int
        Number of subjects set J
    patch_width: int
        Width of patches
    patch_height: int
        Height of patches
    dataI: array
        Patches of data set I
    dataJ:  array
        Patches of data set J

    Returns
    -------
    setI: array
        Selected patches of data set I
    setJ: array
        Selected patches of data set J

    """

    # Preallocate
    setI = np.zeros((nI * num_patch_seq[n], patch_width * patch_height))
    setJ = np.zeros((nJ * num_patch_seq[n], patch_width * patch_height))

    # Set c to 0
    c = 0

    for i in range(nI):
        setI[c: c + num_patch_seq[n], :] = dataI[num_patch_seq[-1] * i: (num_patch_seq[-1] * i) + num_patch_seq[n]]

        # Tick up
        c = c + num_patch_seq[n]

    # Set k to 0
    k = 0

    for j in range(nJ):
        setJ[k: k + num_patch_seq[n], :] = dataJ[num_patch_seq[-1] * j: (num_patch_seq[-1] * j) + num_patch_seq[n]]

        # Tick up
        k = k + num_patch_seq[n]

    return setI, setJ


def set_classifier(X_build, y_build, num_classes, c_range, model, num_folds):

    """
    Create a classifier with optimal regularization parameter

    Parameters
    ----------
    X_build: array
        Training data
    y_build: array
        Training labels
    num_classes: int
        Number of classes to use in neural network
    c_range: array
        Range of possible regularization parameters
    model: str
        Type of model (classifier) to use, options:
        'svm' - support vector machine,
        'lr' - logistic regression,
        'cnn' - convolutional neural network
    num_folds: int
        Number of folds to use in cross-validation
        
    Returns
    -------
    sklearn classifier: trained classifier with a predict function

    """

    # Support vector machine
    if model == 'svm':

        if len(c_range) == 1:
            bestC = c_range[0]

        else:
            # Grid search, find optimal regularization parameter
            modelgs = skms.GridSearchCV(estimator=svm.SVC(kernel='linear',
                                                          class_weight='balanced',
                                                          probability=True),
                                        cv=num_folds, param_grid=dict(C=c_range))

            # Fit grid search model
            modelgs.fit(X_build, y_build)

            # Best regularization parameter C
            bestC = modelgs.best_estimator_.C

        return sksv.SVC(C=bestC, class_weight='balanced', probability=True)

    # Logistic regression
    elif model == 'lr':

        if len(c_range) == 1:
            bestC = c_range[0]

        else:
            # Grid search, find optimal regularization parameter
            modelgs = skms.GridSearchCV(estimator=sklm.LogisticRegression(class_weight='balanced',
                                                                          solver='liblinear'),
                                        cv=num_folds, param_grid=dict(C=c_range))

            # Fit grid search model
            modelgs.fit(X_build, y_build)

            # Best regularization parameter C
            bestC = modelgs.best_estimator_.C

        return sklm.LogisticRegression(C=bestC, class_weight='balanced')

    # Convolutional neural network
    elif model == 'cnn':

        # Start sequential model
        net = km.Sequential()

        # Convolutional part
        net.add(kl.Conv2D(16, kernel_size=(3, 3),
                          activation='relu',
                          padding='valid',
                          kernel_regularizer=kr.l2(0.001),
                          input_shape=(X_build.shape[1], X_build.shape[2], 1)))
        net.add(kl.MaxPooling2D(pool_size=(2, 2), padding='valid'))
        net.add(kl.Dropout(0.2))
        net.add(kl.Flatten())

        # Fully-connected part
        net.add(kl.Dense(32, activation='relu',
                         kernel_regularizer=kr.l2(0.001)))
        net.add(kl.Dropout(0.2))
        net.add(kl.Dense(16, activation='relu',
                         kernel_regularizer=kr.l2(0.001)))
        net.add(kl.Dense(num_classes, activation='softmax'))

        # Compile network architecture
        net.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        return net

    else:
        print('Model format unknown')


def classify(X_build, X_test, y_build, y_test, model, patch_width, patch_height, num_folds, c_range, batch_size,
             num_epochs):

    """
    Classification of patches

    Parameters
    ----------
    X_build: array
        Training data
    X_test: array
        Test data
    y_build: array
        Training labels
    y_test: array
        Test labels
    model: str
        Type of model (classifier) to use, options:
        'svm' - support vector machine,
        'lr' - logistic regression,
        'cnn' - convolutional neural network
    patch_width: int
        Width of patch
    patch_height: int
        Height of patch
    num_folds: int
        Number of folds to use in cross-validation
    c_range: array
        Range of possible regularization parameters
    batch_size: int
        Size of the batches to cut the data set into
    num_epochs: int
        Number of epochs in training a neural network

    Returns
    -------
    error: int
        Classification error
    model: sklearn classifier

    for 'lr' and 'svm'
    prob: array
        Classification probabilities

    for 'cnn'
    pred: array
        Classification predictions

    """

    # Unique number of classes based on training labels
    nClasses = len(np.unique(y_build))

    if model.lower() in ['lr', 'svm']:

        # Model
        model = set_classifier(X_build, y_build, nClasses, c_range=c_range, model=model, num_folds=num_folds)

        # Fit
        model.fit(X_build, y_build)

        # Error
        error = 1 - model.score(X_test, y_test)

        # Probabilities
        prob = model.predict_proba(X_test)

        return error, prob, model

    elif model == 'cnn':

        # Apply to_categorical
        y_train = to_categorical(y_build)
        y_test = to_categorical(y_test)

        # Reshape
        X_train = X_build.reshape(X_build.shape[0], patch_width, patch_height, 1)
        X_test = X_test.reshape(X_test.shape[0], patch_width, patch_height, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # Model
        model = set_classifier(X_train, y_train, nClasses, c_range=c_range, model=model, num_folds=num_folds)

        # Fit
        model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.2, shuffle=True)

        # Error
        error = 1 - model.test_on_batch(X_test, y_test)[1]

        # Class predictions 
        pred = model.predict_classes(X_test)

        return error, pred, model

    else:
        print('Model format unknown')


def shape_beta(prob):

    """
    Fit beta distribution to probabilities and obtain shape parameters

    Parameters
    ----------
    prob: array
        Classification probabilities of patches from test set

    Returns
    -------
    shapeA: int
        Shape parameter A of beta distribution
    shapeB: int
        Shape parameter B of beta distribution
        
    """

    # Check for NA's or infinity in prob
    if np.any(np.isnan(prob)) or np.any(np.isinf(prob)):
        raise ValueError('numerical problems with prob')

    # Fitting Beta distribution to probabilities, R function
    fit = dist(vector(prob), "beta", method="mle")

    # Shape parameters
    param = fit[0]
    param_ = np.array(param)
    shapeA = param_[0]
    shapeB = param_[1]

    return shapeA, shapeB


def CMA23(L):

    """
    https://github.com/wmkouw/mrai-net

    Map CMA's automatic segmentation to {BCK, CSF, GM, WM}
    Sets Brainstem and Cerebellum to background (16=0, 6, 7, 8, 45, 46, 47=0)

    Parameters
    ----------
    L:  array
        Label matrix

    Returns
    -------
    L:  array
        Label matrix 

    """

    # Number of subjects
    nI = L.shape[0]

    # Re-map to
    L = -L
    for i in range(nI):
        L[i][L[i] == -0] = 0
        L[i][L[i] == -1] = 0
        L[i][L[i] == -2] = 3
        L[i][L[i] == -3] = 2
        L[i][L[i] == -4] = 1
        L[i][L[i] == -5] = 1
        L[i][L[i] == -6] = 0
        L[i][L[i] == -7] = 0
        L[i][L[i] == -8] = 0
        L[i][L[i] == -9] = 2
        L[i][L[i] == -10] = 2
        L[i][L[i] == -11] = 2
        L[i][L[i] == -12] = 2
        L[i][L[i] == -13] = 2
        L[i][L[i] == -14] = 1
        L[i][L[i] == -15] = 1
        L[i][L[i] == -16] = 0
        L[i][L[i] == -17] = 2
        L[i][L[i] == -18] = 2
        L[i][L[i] == -19] = 2
        L[i][L[i] == -20] = 2
        L[i][L[i] == -21] = 0
        L[i][L[i] == -22] = 0
        L[i][L[i] == -23] = 2
        L[i][L[i] == -24] = 1
        L[i][L[i] == -25] = 0
        L[i][L[i] == -26] = 2
        L[i][L[i] == -27] = 2
        L[i][L[i] == -28] = 2
        L[i][L[i] == -29] = 0
        L[i][L[i] == -30] = 0
        L[i][L[i] == -31] = 0
        L[i][L[i] == -32] = 0
        L[i][L[i] == -33] = 0
        L[i][L[i] == -34] = 0
        L[i][L[i] == -35] = 0
        L[i][L[i] == -36] = 0
        L[i][L[i] == -37] = 0
        L[i][L[i] == -38] = 0
        L[i][L[i] == -39] = 0
        L[i][L[i] == -40] = 0
        L[i][L[i] == -41] = 3
        L[i][L[i] == -42] = 2
        L[i][L[i] == -43] = 1
        L[i][L[i] == -44] = 1
        L[i][L[i] == -45] = 0
        L[i][L[i] == -46] = 0
        L[i][L[i] == -47] = 0
        L[i][L[i] == -48] = 2
        L[i][L[i] == -49] = 2
        L[i][L[i] == -50] = 2
        L[i][L[i] == -51] = 2
        L[i][L[i] == -52] = 2
        L[i][L[i] == -53] = 2
        L[i][L[i] == -54] = 2
        L[i][L[i] == -55] = 2
        L[i][L[i] == -56] = 2
        L[i][L[i] == -57] = 0
        L[i][L[i] == -58] = 2
        L[i][L[i] == -59] = 2
        L[i][L[i] == -60] = 2
        L[i][L[i] == -61] = 0
        L[i][L[i] == -62] = 0
        L[i][L[i] == -63] = 0
        L[i][L[i] == -64] = 0
        L[i][L[i] == - 65] = 0
        L[i][L[i] == -66] = 0
        L[i][L[i] == -67] = 0
        L[i][L[i] == -68] = 0
        L[i][L[i] == -69] = 0
        L[i][L[i] == -70] = 0
        L[i][L[i] == -71] = 0
        L[i][L[i] == -72] = 0
        L[i][L[i] == -73] = 0
        L[i][L[i] == -74] = 0
        L[i][L[i] == -75] = 0
        L[i][L[i] == -76] = 0
        L[i][L[i] == -77] = 0
        L[i][L[i] == -78] = 0
        L[i][L[i] == -79] = 0
        L[i][L[i] == -80] = 0
        L[i][L[i] == -81] = 0
        L[i][L[i] == -82] = 0
        L[i][L[i] == -83] = 0
        L[i][L[i] == -84] = 3

    return L


def tissue_label(X, idx_mid):

    """
    Extract tissue label
    Tissue label of a patch is based on the middle voxel/pixel of a patch
    Options: CSF (1), GM (2), WM (3)

    Parameters
    ----------
    X:  array
        Patches from which tissue labels are extracted 

    idx_mid: int
        Index of middle of patch, with dimensions pW x pH

    Returns
    -------
    ts: array
        Tissue labels of patches 
        
    """

    # Shape of array X
    sh = np.shape(X)[0]

    # Preallocate
    ts = np.zeros((sh, 1))

    # Obtain tissue label of each patch  
    for n in range(sh):
        ts[n, :] = X[n, idx_mid]

    return ts
