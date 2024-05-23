import os
import sys
import warnings
from glob import glob
from optparse import OptionParser
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm

# Global constants
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.setrecursionlimit(40000)
root_dir = Path(__file__).parent
np.random.seed(42)

# Preprocessing parameters
col_size = [(96, 96, 64), (96, 96, 96), (112, 112, 64), (64, 64, 32)]
input_rows, input_cols, input_depth = (64, 64, 32)
local_col_size = [(32, 32, 16), (16, 16, 16), (32, 32, 32), (8, 8, 8)]
local_input_rows, local_input_cols, local_input_depth = (16, 16, 16)


class setup_config():
    hu_max = 1000.0
    hu_min = -1000.0
    HU_thred = (-150.0 - hu_min) / (hu_max - hu_min)

    def __init__(self,
                 input_rows=None,
                 input_cols=None,
                 input_deps=None,
                 crop_rows=None,
                 crop_cols=None,
                 len_border=None,
                 len_border_z=None,
                 scale=None,
                 DATA_DIR=None,
                 SAVE_DIR=None,
                 train_fold=[0, 8],
                 valid_fold=[],
                 test_fold=[],
                 len_depth=None,
                 lung_min=0.7,
                 lung_max=1.0,
                 ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.train_fold = train_fold
        self.valid_fold = valid_fold
        self.test_fold = test_fold
        self.len_depth = len_depth
        self.lung_min = lung_min
        self.lung_max = lung_max
        self.SAVE_DIR = SAVE_DIR

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class Preprocessor:
    def __init__(self, config):
        self.config = config

    def infinite_generator_from_one_volume(self, img_array, save_dir, name):
        img_array[img_array < self.config.hu_min] = self.config.hu_min
        img_array[img_array > self.config.hu_max] = self.config.hu_max
        img_array = 1.0 * (img_array - self.config.hu_min) / (self.config.hu_max - self.config.hu_min)
        num_pair = 0
        while True:
            crop_window1, crop_window2, local_windows = self.crop_pair(img_array)
            crop_window = np.stack((crop_window1, crop_window2), axis=0)
            # crop_window = np.concatenate([crop_window, local_windows], axis=0)
            # print(crop_window.shape)
            np.save(os.path.join(save_dir, name + '_global_' + str(num_pair) + '.npy'), crop_window)
            np.save(os.path.join(save_dir, name + '_local_' + str(num_pair) + '.npy'), local_windows)
            num_pair += 1
            if num_pair == self.config.scale:
                break

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
                size_index1 = np.random.randint(0, len(col_size))
                crop_rows1, crop_cols1, crop_deps1 = col_size[size_index1]
                size_index2 = np.random.randint(0, len(col_size))
                crop_rows2, crop_cols2, crop_deps2 = col_size[size_index2]
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
                iou = self.cal_iou(box1, box2)
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

            if crop_rows1 != input_rows or crop_cols1 != input_cols or crop_deps1 != input_depth:
                crop_window1 = resize(crop_window1,
                                      (input_rows, input_cols, input_depth + self.config.len_depth),
                                      preserve_range=True,
                                      )
            if crop_rows2 != input_rows or crop_cols2 != input_cols or crop_deps2 != input_depth:
                crop_window2 = resize(crop_window2,
                                      (input_rows, input_cols, input_depth + self.config.len_depth),
                                      preserve_range=True,
                                      )
            t_img1 = np.zeros((input_rows, input_cols, input_depth), dtype=float)
            d_img1 = np.zeros((input_rows, input_cols, input_depth), dtype=float)
            t_img2 = np.zeros((input_rows, input_cols, input_depth), dtype=float)
            d_img2 = np.zeros((input_rows, input_cols, input_depth), dtype=float)
            for d in range(input_depth):
                for i in range(input_rows):
                    for j in range(input_cols):
                        for k in range(self.config.len_depth):
                            if crop_window1[i, j, d + k] >= self.config.HU_thred:
                                t_img1[i, j, d] = crop_window1[i, j, d + k]
                                d_img1[i, j, d] = k
                                break
                            if k == self.config.len_depth - 1:
                                d_img1[i, j, d] = k
            for d in range(input_depth):
                for i in range(input_rows):
                    for j in range(input_cols):
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
                local_size_index = np.random.randint(0, len(local_col_size))
                local_crop_rows, local_crop_cols, local_crop_deps = local_col_size[local_size_index]
                local_window = img_array1[local_x: local_x + local_crop_rows,
                               local_y: local_y + local_crop_cols,
                               local_z: local_z + local_crop_deps
                               ]
                # if local_crop_rows != local_input_rows or local_crop_cols != local_input_cols or local_crop_deps != local_input_depth:
                local_window = resize(local_window,
                                      (local_input_rows, local_input_cols, local_input_depth),
                                      preserve_range=True,
                                      )
                local_windows.append(local_window)
            return crop_window1[:, :, :input_depth], crop_window2[:, :, :input_depth], np.stack(local_windows, axis=0)

    def get_self_learning_data(self):
        brats_path = str(root_dir / self.config.DATA_DIR)
        file_list = glob(os.path.join(brats_path, '*.nii.gz'))
        save_dir = str(root_dir / self.config.SAVE_DIR)
        for i, img_file in enumerate(tqdm(file_list)):
            img_name = os.path.split(img_file)[-1]
            img_array = self.load_sitk_with_resample(img_file)
            img_array = sitk.GetArrayFromImage(img_array)
            img_array = img_array.transpose(2, 1, 0)
            # print(img_array.shape)
            self.infinite_generator_from_one_volume(img_array, save_dir, img_name[:-7])

    def cal_iou(self, box1, box2):
        """
        :param box1: = [xmin1, ymin1, xmax1, ymax1]
        :param box2: = [xmin2, ymin2, xmax2, ymax2]
        :return:
        """
        xmin1, xmax1, ymin1, ymax1, zmin1, zmax1 = box1
        xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = box2
        # 计算每个矩形的面积
        s1 = (xmax1 - xmin1) * (ymax1 - ymin1) * (zmax1 - zmin1)  # C的面积
        s2 = (xmax2 - xmin2) * (ymax2 - ymin2) * (zmax2 - zmin2)  # G的面积

        # 计算相交矩形
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)
        zmin = max(zmin1, zmin2)
        zmax = min(zmax1, zmax2)
        w = max(0, xmax - xmin)
        h = max(0, ymax - ymin)
        d = max(0, zmax - zmin)
        area = w * h * d  # C∩G的面积
        iou = area / (s1 + s2 - area)
        return iou

    def load_sitk_with_resample(self, img_path):
        outsize = [0, 0, 0]
        outspacing = [1, 1, 1]

        vol = sitk.ReadImage(img_path)
        tmp = sitk.GetArrayFromImage(vol)
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
        newvol = resampler.Execute(vol)
        return newvol


def parse_args():
    parser = OptionParser()
    parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
    parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
    parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
    parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=64, type="int")
    parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=64, type="int")
    parser.add_option("--data", dest="data", help="the directory of BraTS dataset", default='BraTS_subset',
                      type="string")
    parser.add_option("--save", dest="save", help="the directory of processed 3D cubes",
                      default='BraTS_preprocessed', type="string")
    parser.add_option("--scale", dest="scale", help="scale of the generator", default=16, type="int")
    (options, _) = parser.parse_args()

    return options


def create_config(options):
    config = setup_config(input_rows=options.input_rows,
                          input_cols=options.input_cols,
                          input_deps=options.input_deps,
                          crop_rows=options.crop_rows,
                          crop_cols=options.crop_cols,
                          scale=options.scale,
                          len_border=70,
                          len_border_z=15,
                          len_depth=3,
                          lung_min=0.7,
                          lung_max=0.15,
                          DATA_DIR=options.data,
                          SAVE_DIR=options.save
                          )
    config.display()

    return config


def run_preprocessing():
    options = parse_args()
    config = create_config(options)

    pp = Preprocessor(config)
    pp.get_self_learning_data()


if __name__ == "__main__":
    run_preprocessing()
