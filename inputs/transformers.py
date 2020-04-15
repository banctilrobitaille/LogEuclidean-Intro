from __future__ import print_function

import math

import nibabel as nib
import nrrd
import numpy as np
import operator
import os
import random
import torch
import warnings
from functools import reduce
from numba import guvectorize

from inputs import Image, ImageType

CHANNEL, DEPTH, HEIGHT, WIDTH = 0, 1, 2, 3


class ToNDTensor(object):
    """
    Creates a torch.Tensor object from a numpy array.
    The transformer supports 3D and 4D numpy arrays. The numpy arrays are transposed in order to create tensors with
    dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
    The dimensions are D: Depth, H: Height, W: Width, C: Channels.
    """

    # noinspection PyArgumentList
    def __call__(self, nd_array):
        """
        :param nd_array: A 3D or 4D numpy array to convert to torch.Tensor
        :return: A torch.Tensor of size (DxHxW) or (CxDxHxW)"""
        if not isinstance(nd_array, np.ndarray):
            raise TypeError("Only {} are supporter".format(np.ndarray))

        if nd_array.ndim == 3:
            nd_tensor = torch.Tensor(nd_array.reshape(nd_array.shape + (1,)))
        elif nd_array.ndim == 4:
            nd_tensor = torch.Tensor(nd_array)
        else:
            raise NotImplementedError("Only 3D or 4D arrays are supported")

        return nd_tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNiftiFile(object):
    """
    Creates a Nifti1Image from a given numpy ndarray
    The numpy arrays are transposed to respect the standard Nifti dimensions (WxHxDxC)
    """

    def __init__(self, file_path, affine):
        self._file_path = file_path
        self._affine = affine

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        output_dir = os.path.dirname(self._file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if nd_array.shape[0] not in [6, 9]:
            nd_array = np.squeeze(nd_array, axis=0)
        else:
            nd_array = np.moveaxis(nd_array, 0, 3)

        nifti1_file = nib.Nifti1Image(nd_array, self._affine)
        nib.save(nifti1_file, self._file_path)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNrrdFile(object):
    """
    Create a .NRRD file and save it at the given path.
    The numpy arrays are transposed to respect the standard NRRD dimensions (WxHxDxC)
    """

    def __init__(self, file_path):
        self._file_path = file_path

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) {} are supported".format(np.ndarray))

        output_dir = os.path.dirname(self._file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        header = self._create_header_from(nd_array)
        nrrd.write(self._file_path, np.moveaxis(nd_array, 0, 3), header=header)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def _create_header_from(nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) {} are supported".format(np.ndarray))

        return {
            'type': nd_array.dtype,
            'dimension': nd_array.ndim,
            'sizes': nd_array.shape,
            'kinds': ['domain', 'domain', 'domain', '3D-matrix'] if nd_array.ndim == 4 else ['domain', 'domain',
                                                                                             'domain'],
            'endian': 'little',
            'encoding': 'raw'
        }


class ToNumpyArray(object):
    """
    Creates a numpy ndarray from a given Nifti or NRRD image file path.
    The numpy arrays are transposed to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
    """

    def __call__(self, image_path):
        if Image.is_nifti(image_path):
            nifti_image = nib.load(image_path)
            nd_array = nifti_image.get_fdata().__array__()
            affine = nifti_image._affine
        elif Image.is_nrrd(image_path):
            nd_array, header = nrrd.read(image_path)
        else:
            raise NotImplementedError("Only {} files are supported !".format(ImageType.ALL))

        if nd_array.ndim == 3:
            nd_array = np.moveaxis(np.expand_dims(nd_array, 3), 3, 0)
        elif nd_array.ndim == 4:
            nd_array = np.moveaxis(nd_array, 3, 0)

        return nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToUniqueTensorValues(object):
    UNIQUE_TENSOR_VALUES_INDEX = [0, 1, 2, 4, 5, 8]
    """
    Creates a numpy ndarray from a given Nifti or NRRD image file path.
    The numpy arrays are transposed to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
    """

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or nd_array.ndim is not 4 or nd_array.shape[0] != 9:
            raise TypeError("Only 4D (CxDxHxW) {} are with 9 channels are supported".format(np.ndarray))

        return nd_array[self.UNIQUE_TENSOR_VALUES_INDEX, :, :, :]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToLogEuclidean(object):
    """
    Convert a DTI image in the Log-Euclidean space.
    To convert the DTI image into the Log-Euclidean space, the eigen-decomposition of each tensor is performed and the
    log of the eigen-values is computed.
    It can mathematically be expressed as follow: log(D) = Ulog(V)U.T where D is a tensor, U is a matrix of eigen-vector
    and V a diagonal matrix of eigen-values.
    Based on: Arsigny, V., Fillard, P., Pennec, X., & Ayache, N. (2006). Log-Euclidean metrics for fast and
    simple calculus on diffusion tensors https://www.ncbi.nlm.nih.gov/pubmed/16788917
    """

    def __call__(self, nd_array):
        """
        :param nd_array: The DTI image as a nd array of dimension CxDxHxW)
        :return: he DTI image in the log-Euclidean space
        """
        warnings.filterwarnings('ignore')
        if not isinstance(nd_array, np.ndarray) or nd_array.ndim is not 4 or nd_array.shape[0] != 9:
            raise TypeError("Only 4D (CxDxHxW) {} are with 9 channels are supported".format(np.ndarray))

        image_as_vector = nd_array.reshape((3, 3, reduce(operator.mul, nd_array.shape[1:], 1)))

        return self.apply(image_as_vector, np.zeros(image_as_vector.shape, dtype='float32')).reshape(nd_array.shape)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    @guvectorize(["void(float64[:,:,:], float64[:,:,:])"], "(m,m,n) -> (m,m,n)", nopython=True)
    def apply(image_vector, output):
        index = 0

        while index < image_vector.shape[2]:
            diffusion_tensor = image_vector[:, :, index]

            # Does not convert the background tensors to log-euclidean
            if np.any(diffusion_tensor):
                eig_val, eig_vec = np.linalg.eigh(diffusion_tensor)
                output[:, :, index] = np.dot(np.dot(np.ascontiguousarray(eig_vec), np.diag(np.log(eig_val))),
                                             np.ascontiguousarray(np.linalg.inv(eig_vec)))
            else:
                output[:, :, index] = diffusion_tensor

            index = index + 1

    @staticmethod
    @guvectorize(["void(float32[:,:,:], float32[:,:,:])"], "(m,m,n) -> (m,m,n)", nopython=True)
    def undo(image_vector, output):
        index = 0

        while index < image_vector.shape[2]:
            log_euclidean_diffusion_tensor = image_vector[:, :, index]

            # Due to noise, negative eigenvalues can arise. Those noisy tensors cannot be converted back to Euclidean.
            if np.any(log_euclidean_diffusion_tensor) and not np.isnan(log_euclidean_diffusion_tensor).any():
                eig_val, eig_vec = np.linalg.eigh(log_euclidean_diffusion_tensor)
                output[:, :, index] = np.dot(np.dot(np.ascontiguousarray(eig_vec), np.diag(np.exp(eig_val))),
                                             np.ascontiguousarray(np.linalg.inv(eig_vec)))
            else:
                output[:, :, index] = log_euclidean_diffusion_tensor

            index = index + 1


class InterpolateNSDTensors(object):
    """
    Interpolates Negative Semi-Definite tensors using trilinear interpolation.
    It computed a weighted sum of the NSD tensors' neighbors in the Log-Euclidean domain.
    """

    def __call__(self, log_euclidean_nd_array):
        if not isinstance(log_euclidean_nd_array, np.ndarray) or log_euclidean_nd_array.ndim is not 4:
            raise TypeError("Only {} are supported".format(np.ndarray.dtype))

        d_index, h_index, w_index = np.where(np.isnan(log_euclidean_nd_array[-1, :, :, :]))

        for index in list(zip(d_index, h_index, w_index)):
            neighbors = self._get_tri_linear_neighbors_and_weights(index, log_euclidean_nd_array)
            log_euclidean_nd_array[:, index[0], index[1], index[2]] = np.dot(np.array(neighbors[0]).T,
                                                                             neighbors[1] / np.sum(neighbors[1]))

        return log_euclidean_nd_array

    def _get_tri_linear_neighbors_and_weights(self, nsd_index, log_euclidean_nd_array):
        """
        Gets the 8 neighbors of the NSD tensors from which to interpolate. The weight associated with each neighbor
        is inversely proportional to the distance between the interpolated tensor and the neighbor.
        :param nsd_index: The index of the NSD tensor.
        :param log_euclidean_nd_array: The log euclidean image as numpy ndarray
        :return: A list of the 8 corner neighbors and their associated weights in separated tuples.
        """
        front, left, down = -1, -1, -1
        back, right, up = 1, 1, 1
        directions = [(front, left, down), (front, left, up), (back, left, down), (back, left, up),
                      (front, right, down), (front, right, up), (back, right, up), (back, right, down)]

        neighbors_and_weights = list(map(lambda direction:
                                         self._get_closest_neighbor_of(log_euclidean_nd_array, nsd_index, direction),
                                         directions))

        return list(zip(*neighbors_and_weights))

    @staticmethod
    def _get_closest_neighbor_of(log_euclidean_nd_array, nsd_index, direction):
        """
        Gets the closest non-NSD tensor to the nsd_index and its weight following a given direction.
        The associated weight is 1/distance, where the distance is the distance from the neighbor and the nsd_index.
        :param log_euclidean_nd_array: The log-euclidean image as ndarray.
        :param nsd_index: The index of the NSD tensor to interpolate.
        :param direction: The direction in which the neighbor is searched.
        :return: The closest neighbor as a 9 values vector and its associated weight.
        """
        distance = 1
        neighbor = None

        try:
            while neighbor is None:
                d, h, w = tuple(((np.array(direction) * distance) + nsd_index))

                if 0 < d < log_euclidean_nd_array.shape[1] and 0 < h < log_euclidean_nd_array.shape[2] and 0 < w < \
                        log_euclidean_nd_array.shape[3]:
                    potential_neighbor = log_euclidean_nd_array[:, d, h, w]
                else:
                    raise IndexError

                if not np.isnan(potential_neighbor).any():
                    neighbor = potential_neighbor
                else:
                    distance = distance + 1

            weight = 1 / distance
        except IndexError:
            neighbor = np.zeros(log_euclidean_nd_array.shape[0])
            weight = 0

        return neighbor, weight

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropToContent(object):
    """
    Crops the image to its content.
    The content's bounding box is defined by the first non-zero slice in each direction (D, H, W)
    """

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        c, d_min, d_max, h_min, h_max, w_min, w_max = self.extract_content_bounding_box_from(nd_array)

        return nd_array[:, d_min:d_max, h_min:h_max, w_min:w_max] if nd_array.ndim is 4 else \
            nd_array[d_min:d_max, h_min:h_max, w_min:w_max]

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def extract_content_bounding_box_from(nd_array):
        """
        Computes the D, H, W min and max values defining the content bounding box.
        :param nd_array: The input image as a numpy ndarray
        :return: The D, H, W min and max values of the bounding box.
        """

        depth_slices = np.any(nd_array, axis=(2, 3))
        height_slices = np.any(nd_array, axis=(1, 3))
        width_slices = np.any(nd_array, axis=(1, 2))

        d_min, d_max = np.where(depth_slices)[1][[0, -1]]
        h_min, h_max = np.where(height_slices)[1][[0, -1]]
        w_min, w_max = np.where(width_slices)[1][[0, -1]]

        return nd_array.shape[CHANNEL], d_min, d_max, h_min, h_max, w_min, w_max


class PadToShape(object):
    def __init__(self, target_shape, padding_value=0, isometric=False):
        self._padding_value = padding_value

        if isometric:
            largest_dimension = max(target_shape[DEPTH], target_shape[WIDTH])

            self._target_shape = (target_shape[CHANNEL], largest_dimension, target_shape[HEIGHT], largest_dimension)
        else:
            self._target_shape = target_shape

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")
        elif nd_array.ndim is not len(self._target_shape):
            raise ValueError(
                "The input image and target shape's dimension does not match {} vs {}".format(nd_array.ndim,
                                                                                              len(self._target_shape)))

        return self.apply(nd_array, self._target_shape, self._padding_value)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def apply(nd_array, target_shape, padding_value):
        deltas = tuple(max(0, target - current) for target, current in zip(target_shape, nd_array.shape))

        if nd_array.ndim == 3:
            nd_array = np.pad(nd_array, ((math.floor(deltas[0] / 2), math.ceil(deltas[0] / 2)),
                                         (math.floor(deltas[1] / 2), math.ceil(deltas[1] / 2)),
                                         (math.floor(deltas[2] / 2), math.ceil(deltas[2] / 2))),
                              'constant', constant_values=padding_value)
        elif nd_array.ndim == 4:
            nd_array = np.pad(nd_array, ((0, 0),
                                         (math.floor(deltas[1] / 2), math.ceil(deltas[1] / 2)),
                                         (math.floor(deltas[2] / 2), math.ceil(deltas[2] / 2)),
                                         (math.floor(deltas[3] / 2), math.ceil(deltas[3] / 2))),
                              'constant', constant_values=padding_value)
        return nd_array

    @staticmethod
    def undo(nd_array, original_shape):
        deltas = tuple(max(0, current - target) for target, current in zip(original_shape, nd_array.shape))

        if nd_array.ndim == 3:
            nd_array = nd_array[
                       math.floor(deltas[0] / 2):-math.ceil(deltas[0] / 2),
                       math.floor(deltas[1] / 2):-math.ceil(deltas[1] / 2),
                       math.floor(deltas[2] / 2):-math.ceil(deltas[2] / 2)]
        elif nd_array.ndim == 4:
            nd_array = nd_array[
                       :,
                       math.floor(deltas[1] / 2):-math.ceil(deltas[1] / 2),
                       math.floor(deltas[2] / 2):-math.ceil(deltas[2] / 2),
                       math.floor(deltas[3] / 2):-math.ceil(deltas[3] / 2)]
        return nd_array


class RandomFlip(object):
    def __init__(self, exec_probability):
        self._exec_probability = exec_probability

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        for axis in (0, 1, 2):
            if random.uniform(0, 1) <= self._exec_probability:
                nd_array = self.apply(nd_array, [axis])

        return nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def apply(nd_array, axes):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        for axis in axes:
            if nd_array.ndim is 3:
                nd_array = np.flip(nd_array, axis)
            else:
                channels = [np.flip(nd_array[c], axis) for c in range(nd_array.shape[0])]
                nd_array = np.stack(channels, axis=0)

        return nd_array

    @staticmethod
    def undo(nd_array, axes):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        for axis in axes[::-1]:
            if nd_array.ndim is 3:
                nd_array = np.flip(nd_array, axis)
            else:
                channels = [np.flip(nd_array[c], axis) for c in range(nd_array.shape[0])]
                nd_array = np.stack(channels, axis=0)

        return nd_array


class RandomRotate90(object):
    def __init__(self, exec_probability):
        self._exec_probability = exec_probability

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        if random.uniform(0, 1) <= self._exec_probability:
            num_rotation = random.randint(0, 4)
            nd_array = self.apply(nd_array, num_rotation)

        return nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def apply(nd_array, num_rotation):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        if nd_array.ndim == 3:
            nd_array = np.rot90(nd_array, num_rotation, (1, 2))
        else:
            channels = [np.rot90(nd_array[c], num_rotation, (1, 2)) for c in range(nd_array.shape[0])]
            nd_array = np.stack(channels, axis=0)

        return nd_array

    @staticmethod
    def undo(nd_array, num_rotation):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        if nd_array.ndim == 3:
            nd_array = np.rot90(nd_array, num_rotation, (2, 1))
        else:
            channels = [np.rot90(nd_array[c], num_rotation, (2, 1)) for c in range(nd_array.shape[0])]
            nd_array = np.stack(channels, axis=0)

        return nd_array


class Normalize(object):
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray):
            raise TypeError("Only  ndarrays are supported")

        return self.apply(nd_array, self._mean, self._std)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def apply(nd_array, mean, std):
        if not isinstance(nd_array, np.ndarray):
            raise TypeError("Only  ndarrays are supported")

        return (nd_array - mean) / std

    @staticmethod
    def undo(nd_array, mean, std):
        if not isinstance(nd_array, np.ndarray):
            raise TypeError("Only  ndarrays are supported")

        return (nd_array * std) + mean


class Flip(object):
    def __init__(self, axis):
        self._axis = axis

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray):
            raise TypeError("Only  ndarrays are supported")

        return self.apply(nd_array, self._axis)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def apply(nd_array, axis):
        if not isinstance(nd_array, np.ndarray):
            raise TypeError("Only  ndarrays are supported")

        return np.flip(nd_array, axis).copy()

    @staticmethod
    def undo(nd_array, axis):
        if not isinstance(nd_array, np.ndarray):
            raise TypeError("Only  ndarrays are supported")

        return np.flip(nd_array, axis).copy()


class TensorFlip(object):
    def __init__(self, axis):
        self._axis = axis

    def __call__(self, tensor):
        return self.apply(tensor, self._axis)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def apply(tensor, axis):
        return tensor.flip(axis)

    @staticmethod
    def undo(tensor, axis):
        return tensor.flip(axis)
