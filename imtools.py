import os
from PIL import Image
from numpy import *


def get_imlist(path):
    """ Returns a list of filenames for
    all jpg images in a directory. """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def imresize(im, sz):
    """ Resize an image array using PIL. """
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))


def histeq(im, nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # get image histogram
    imhist, bins = histogram(im.flatten(), nbr_bins, density=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


def compute_average(imlist):
    """ Compute the average of a list of images. """
    # open first image and make into array of type float
    averageim = array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname + '...skipped')
    averageim /= len(imlist)
    # return average as uint8
    return array(averageim, 'uint8')


def pca(X):
    """ Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance and mean.
    """
    # get dimensions
    num_data, dim = X.shape
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    if dim > num_data:
        # PCA - compact trick used
        M = dot(X, X.T)  # covariance matrix
        e, EV = linalg.eigh(M)  # eigenvalues and eigenvectors
        tmp = dot(X.T, EV).T  # this is the compact trick
        V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA - SVD used
        U, S, V = linalg.svd(X)
        V = V[:num_data]  # only makes sense to return the first num_data
    # return the projection matrix, the variance and the mean
    return V, S, mean_X


""" Here is what the saving and loading above looks like using with() .
    # open file and save
        with open(’font_pca_modes.pkl’, ’wb’) as f:
            pickle.dump(immean,f)
            pickle.dump(V,f)
    and
    # open file and load
        with open(’font_pca_modes.pkl’, ’rb’) as f:
            immean = pickle.load(f)
            V = pickle.load(f)
This might look strange the first time you see it but it is a very useful construct. If you
don’t like it, just use the open and close functions as above.

As an alternative to using pickle, NumPy also has simple functions for reading and
writing text files that can be useful if your data does not contain complicated struc-
tures, for example a list of points clicked in an image. To save an array x to file use
        savetxt(’test.txt’,x,’%i’)
The last parameter indicates that integer format should be used. Similarly, reading is
done like this:
        x = loadtxt(’test.txt’)
You can find out more from the online documentation 
http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html.
Lastly, NumPy has dedicated functions for saving and loading arrays. Look for save()
and load() in the online documentation for the details. """