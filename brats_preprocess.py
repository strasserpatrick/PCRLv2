import os
import sys
from dataclasses import dataclass
from glob import glob
from optparse import OptionParser
from pathlib import Path
from pprint import pprint

import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

# Global constants
sys.setrecursionlimit(40000)
root_dir = Path(__file__).parent
np.random.seed(42)

# Preprocessing parameters
col_size_sampling_variants = [(96, 96, 64), (96, 96, 96), (112, 112, 64), (64, 64, 32)]
local_col_size_sampling_variants = [(32, 32, 16), (16, 16, 16), (32, 32, 32), (8, 8, 8)]


@dataclass
class PreprocessingConfig:
    """
    Configuration for the preprocessing
    TODO: add descriptions for each parameter
    """
    input_rows: int
    input_cols: int
    input_depth: int
    crop_rows: int
    crop_cols: int
    scale: float
    DATA_DIR: str
    SAVE_DIR: str
    local_input_rows: int = 16
    local_input_cols: int = 16
    local_input_depth: int = 16
    hu_max: float = 1000.0
    hu_min: float = -1000.0
    HU_thred: float = (-150.0 - hu_min) / (hu_max - hu_min)
    len_border: int = 70
    len_border_z: int = 15
    len_depth: int = 3
    lung_min: float = 0.7
    lung_max: float = 0.15


class PCRLv2Preprocessor:
    def __init__(self, config):
        self.config = config

    def local_global_3d_cube_generator(self, img_array, save_dir, file_name):
        """
        Generates 3D cubes from the image and saves them to the disk
        These consist of a global cube and 6 local cubes

        :param img_array: np.array, 3D image
        :param save_dir: str, path to save directory
        :param file_name: str, name of the file

        :return: None
        """
        # Normalize HU values
        img_array[img_array < self.config.hu_min] = self.config.hu_min
        img_array[img_array > self.config.hu_max] = self.config.hu_max
        img_array = 1.0 * (img_array - self.config.hu_min) / (self.config.hu_max - self.config.hu_min)

        # generate 3d pairs
        for i in range(self.config.scale):
            crop_window1, crop_window2, local_windows = self.crop_pair(img_array)
            crop_window = np.stack((crop_window1, crop_window2), axis=0)
            np.save(os.path.join(save_dir, file_name + '_global_' + str(i) + '.npy'), crop_window)
            np.save(os.path.join(save_dir, file_name + '_local_' + str(i) + '.npy'), local_windows)

    def crop_pair(self, img_array):
        while True:
            size_x, size_y, size_z = img_array.shape
            # print(img_array.shape)
            img_array1 = img_array.copy()
            img_array2 = img_array.copy()
            if size_z - 64 - self.config.len_depth - 1 - self.config.len_border_z < self.config.len_border_z:
                pad = size_z - 64 - self.config.len_depth - 1 - self.config.len_border_z - self.config.len_border_z
                padding = [0, 0, -pad + 1]
                img_array1 = np.pad(img_array1, padding, mode='constant', constant_values=0)

            if size_z - 64 - self.config.len_depth - 1 - self.config.len_border_z < self.config.len_border_z:
                pad = size_z - 64 - self.config.len_depth - 1 - self.config.len_border_z - self.config.len_border_z
                padding = [0, 0, -pad + 1]
                img_array2 = np.pad(img_array2, padding, mode='constant', constant_values=0)
                size_z += -pad + 1
            while True:
                size_index1 = np.random.randint(0, len(col_size_sampling_variants))
                crop_rows1, crop_cols1, crop_deps1 = col_size_sampling_variants[size_index1]
                size_index2 = np.random.randint(0, len(col_size_sampling_variants))
                crop_rows2, crop_cols2, crop_deps2 = col_size_sampling_variants[size_index2]
                if size_x - crop_rows1 - 1 - self.config.len_border <= self.config.len_border:
                    crop_rows1 -= 32
                    crop_cols1 -= 32
                if size_x - crop_rows2 - 1 - self.config.len_border <= self.config.len_border:
                    crop_rows2 -= 32
                    crop_cols2 -= 32
                start_x1 = np.random.randint(0 + self.config.len_border,
                                             size_x - crop_rows1 - 1 - self.config.len_border)
                start_y1 = np.random.randint(0 + self.config.len_border,
                                             size_y - crop_cols1 - 1 - self.config.len_border)
                start_z1 = np.random.randint(0 + self.config.len_border_z,
                                             size_z - crop_deps1 - self.config.len_depth - 1 - self.config.len_border_z)
                start_x2 = np.random.randint(0 + self.config.len_border,
                                             size_x - crop_rows2 - 1 - self.config.len_border)
                start_y2 = np.random.randint(0 + self.config.len_border,
                                             size_y - crop_cols2 - 1 - self.config.len_border)
                start_z2 = np.random.randint(0 + self.config.len_border_z,
                                             size_z - crop_deps2 - self.config.len_depth - 1 - self.config.len_border_z)
                box1 = (
                    start_x1, start_x1 + crop_rows1, start_y1, start_y1 + crop_cols1, start_z1, start_z1 + crop_deps1)
                box2 = (
                    start_x2, start_x2 + crop_rows2, start_y2, start_y2 + crop_cols2, start_z2, start_z2 + crop_deps2)
                iou = self.calculate_iou(box1, box2)
                # print(iou, start_x1, start_y1, start_z1, start_x2, start_y2, start_z2)
                if iou > 0.3:
                    break

            crop_window1 = img_array1[start_x1: start_x1 + crop_rows1,
                           start_y1: start_y1 + crop_cols1,
                           start_z1: start_z1 + crop_deps1 + self.config.len_depth,
                           ]

            crop_window2 = img_array2[start_x2: start_x2 + crop_rows2,
                           start_y2: start_y2 + crop_cols2,
                           start_z2: start_z2 + crop_deps2 + self.config.len_depth,
                           ]

            if crop_rows1 != self.config.input_rows or crop_cols1 != self.config.input_cols or crop_deps1 != self.config.input_depth:
                crop_window1 = resize(crop_window1,
                                      (self.config.input_rows, self.config.input_cols,
                                       self.config.input_depth + self.config.len_depth),
                                      preserve_range=True,
                                      )
            if crop_rows2 != self.config.input_rows or crop_cols2 != self.config.input_cols or crop_deps2 != self.config.input_depth:
                crop_window2 = resize(crop_window2,
                                      (self.config.input_rows, self.config.input_cols,
                                       self.config.input_depth + self.config.len_depth),
                                      preserve_range=True,
                                      )
            t_img1 = np.zeros((self.config.input_rows, self.config.input_cols, self.config.input_depth), dtype=float)
            d_img1 = np.zeros((self.config.input_rows, self.config.input_cols, self.config.input_depth), dtype=float)
            t_img2 = np.zeros((self.config.input_rows, self.config.input_cols, self.config.input_depth), dtype=float)
            d_img2 = np.zeros((self.config.input_rows, self.config.input_cols, self.config.input_depth), dtype=float)
            for d in range(self.config.input_depth):
                for i in range(self.config.input_rows):
                    for j in range(self.config.input_cols):
                        for k in range(self.config.len_depth):
                            if crop_window1[i, j, d + k] >= self.config.HU_thred:
                                t_img1[i, j, d] = crop_window1[i, j, d + k]
                                d_img1[i, j, d] = k
                                break
                            if k == self.config.len_depth - 1:
                                d_img1[i, j, d] = k
            for d in range(self.config.input_depth):
                for i in range(self.config.input_rows):
                    for j in range(self.config.input_cols):
                        for k in range(self.config.len_depth):
                            if crop_window2[i, j, d + k] >= self.config.HU_thred:
                                t_img2[i, j, d] = crop_window2[i, j, d + k]
                                d_img2[i, j, d] = k
                                break
                            if k == self.config.len_depth - 1:
                                d_img2[i, j, d] = k

            d_img1 = d_img1.astype('float32')
            d_img1 /= (self.config.len_depth - 1)
            d_img1 = 1.0 - d_img1
            d_img2 = d_img2.astype('float32')
            d_img2 /= (self.config.len_depth - 1)
            d_img2 = 1.0 - d_img2

            if np.sum(d_img1) > self.config.lung_max * crop_cols1 * crop_deps1 * crop_rows1:
                continue
            # print(np.sum(d_img1))
            if np.sum(d_img2) > self.config.lung_max * crop_cols1 * crop_deps1 * crop_rows1:
                continue
            # we start to crop the local windows
            x_min = min(box1[0], box2[0])
            x_max = max(box1[1], box2[1])
            y_min = min(box1[2], box2[2])
            y_max = max(box1[3], box2[3])
            z_min = min(box1[4], box2[4])
            z_max = max(box1[5], box2[5])
            local_windows = []
            for i in range(6):
                local_x = np.random.randint(max(x_min - 3, 0), min(x_max + 3, size_x))
                local_y = np.random.randint(max(y_min - 3, 0), min(y_max + 3, size_y))
                local_z = np.random.randint(max(z_min - 3, 0), min(z_max + 3, size_z))
                local_size_index = np.random.randint(0, len(local_col_size_sampling_variants))
                local_crop_rows, local_crop_cols, local_crop_deps = local_col_size_sampling_variants[local_size_index]
                local_window = img_array1[local_x: local_x + local_crop_rows,
                               local_y: local_y + local_crop_cols,
                               local_z: local_z + local_crop_deps
                               ]
                # if local_crop_rows != local_input_rows or local_crop_cols != local_input_cols or local_crop_deps != local_input_depth:
                local_window = resize(local_window,
                                      (self.config.local_input_rows, self.config.local_input_cols,
                                       self.config.local_input_depth),
                                      preserve_range=True,
                                      )
                local_windows.append(local_window)
            return crop_window1[:, :, :self.config.input_depth], crop_window2[:, :, :self.config.input_depth], np.stack(
                local_windows, axis=0)

    def get_self_learning_data(self):
        """
        Main function to process all images in the dataset

        :return: None
        """
        brats_path = str(root_dir / self.config.DATA_DIR)
        file_list = glob(os.path.join(brats_path, '*.nii.gz'))
        save_dir = str(root_dir / self.config.SAVE_DIR)
        for i, img_file in enumerate(tqdm(file_list)):
            self.process_image(img_file, save_dir)

    def process_image(self, img_file, save_dir):
        """
        Loads image from file, resamples it and generates 3D cubes

        :param img_file: str, path to image file
        :param save_dir: str, path to save directory

        :return: None
        """
        img_name = os.path.split(img_file)[-1]
        img_array = self.load_sitk_with_resample(img_file)
        img_array = sitk.GetArrayFromImage(img_array)
        img_array = img_array.transpose(2, 1, 0)
        self.local_global_3d_cube_generator(img_array, save_dir, img_name[:-7])

    def calculate_iou(self, box1, box2):
        """
        Calculates Intersection over Union for two boxes

        :param box1: tuple, coordinates of the first box
        :param box2: tuple, coordinates of the second box

        :return: float, Intersection over Union value
        """

        # unpack the coordinates from boxes
        x_start_box1, x_end_box1, y_start_box1, y_end_box1, z_start_box1, z_end_box1 = box1
        x_start_box2, x_end_box2, y_start_box2, y_end_box2, z_start_box2, z_end_box2 = box2

        # compute the volume of boxes
        area_box1 = (x_end_box1 - x_start_box1) * (y_end_box1 - y_start_box1) * (z_end_box1 - z_start_box1)
        area_box2 = (x_end_box2 - x_start_box2) * (y_end_box2 - y_start_box2) * (z_end_box2 - z_start_box2)

        # find the intersection box and compute the volume
        x_min = max(x_start_box1, x_start_box2)
        y_min = max(y_start_box1, y_start_box2)
        x_max = min(x_end_box1, x_end_box2)
        y_max = min(y_end_box1, y_end_box2)
        z_min = max(z_start_box1, z_start_box2)
        z_max = min(z_end_box1, z_end_box2)

        intersection_w = max(0, x_max - x_min)
        intersection_h = max(0, y_max - y_min)
        intersection_d = max(0, z_max - z_min)
        intersection_area = intersection_w * intersection_h * intersection_d

        # compute the intersection over union
        iou = intersection_area / (area_box1 + area_box2 - intersection_area)
        return iou

    def load_sitk_with_resample(self, img_path):
        """
        Loads image from file and resamples it, so that the spacing is 1x1x1
        This means that the image will be isotropic.

        :param img_path: str, path to image file

        :return: SimpleITK image with isotropic spacing of 1x1x1
        """

        outsize = [0, 0, 0]
        outspacing = [1, 1, 1]

        vol = sitk.ReadImage(img_path)
        inputsize = vol.GetSize()
        inputspacing = vol.GetSpacing()

        transform = sitk.Transform()
        transform.SetIdentity()

        outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
        outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
        outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)

        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(vol.GetOrigin())
        resampler.SetOutputSpacing(outspacing)
        resampler.SetOutputDirection(vol.GetDirection())
        resampler.SetSize(outsize)
        new_vol = resampler.Execute(vol)
        return new_vol


def parse_args():
    parser = OptionParser()
    parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
    parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
    parser.add_option("--input_depth", dest="input_depth", help="input depth", default=32, type="int")
    parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=64, type="int")
    parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=64, type="int")
    parser.add_option("--data", dest="data", help="the directory of BraTS dataset", default='BraTS_subset',
                      type="string")
    parser.add_option("--save", dest="save", help="the directory of processed 3D cubes",
                      default='BraTS_preprocessed', type="string")
    parser.add_option("--scale", dest="scale", help="scale of the generator", default=16, type="int")
    (options, _) = parser.parse_args()

    config = PreprocessingConfig(
        input_rows=options.input_rows,
        input_cols=options.input_cols,
        input_depth=options.input_depth,
        crop_rows=options.crop_rows,
        crop_cols=options.crop_cols,
        scale=options.scale,
        DATA_DIR=options.data,
        SAVE_DIR=options.save
    )

    return config


def run_preprocessing():
    config = parse_args()

    print("Preprocessing configuration:")
    pprint(config)

    pp = PCRLv2Preprocessor(config)
    pp.get_self_learning_data()


if __name__ == "__main__":
    run_preprocessing()
