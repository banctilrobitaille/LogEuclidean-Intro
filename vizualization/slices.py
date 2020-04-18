from enum import Enum

import numpy as np
import seaborn as sns
from kerosene.utils.constants import EPSILON
import matplotlib.pyplot as plt


class SliceType(Enum):
    SAGITAL = "Sagital"
    CORONAL = "Coronal"
    AXIAL = "Axial"

    def __str__(self):
        return self.value


class ImageSlicer(object):

    @staticmethod
    def get_slice(image: np.array, slice_type: SliceType, normalize=False, MNI_space=False):
        if slice_type == SliceType.SAGITAL:
            if not MNI_space:
                slice = np.rot90(image[..., int(image.shape[1] / 2), :, :], axes=(1, 2))
            else:
                slice = np.rot90(image[..., int(image.shape[1] / 2), :, :], axes=(1, 2))
        elif slice_type == SliceType.CORONAL:
            if not MNI_space:
                slice = np.rot90(image[..., int(image.shape[2] / 2), :], axes=(1, 2))
            else:
                slice = np.rot90(image[..., int(image.shape[2] / 2), :], axes=(1, 2))
        elif slice_type == SliceType.AXIAL:
            if not MNI_space:
                slice = np.rot90(image[..., int(image.shape[3] / 2)], axes=(1, 2), k=-1)
            else:
                slice = np.rot90(image[..., int(image.shape[3] / 2)], axes=(1, 2))
        else:
            raise NotImplementedError("The provided slice type ({}) not found.".format(slice_type))

        return ImageSlicer.normalize(slice) if normalize else slice

    @staticmethod
    def normalize(image: np.array):
        return (image - np.min(image)) / (np.ptp(image) + EPSILON)


def plot_slices(sagital_slice: np.array, coronal_slice: np.array, axial_slice: np.array, cmap="viridis"):
    sns.set(font_scale=2)
    fig, ax = plt.subplots(1, 3, figsize=(30, 30))
    ax[0].imshow(sagital_slice, cmap=cmap)
    ax[0].set_title("Sagital")
    ax[0].grid(None)
    ax[1].imshow(coronal_slice, cmap=cmap)
    ax[1].set_title("Coronal")
    ax[1].grid(None)
    ax[2].imshow(axial_slice, cmap=cmap)
    ax[2].set_title("Axial")
    ax[2].grid(None)
