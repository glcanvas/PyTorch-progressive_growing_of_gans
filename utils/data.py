# -*- coding: utf-8 -*-
import os
from glob import glob
import numpy as np 
import h5py

"""
def get_img(img_path, is_crop=True, crop_h=256, resize_h=64, normalize=False):
    img = scipy.misc.imread(img_path, mode='RGB').astype(np.float)
    resize_w = resize_h
    if is_crop:
        crop_w = crop_h
        h, w = img.shape[:2]
        j = int(round((h - crop_h)/2.))
        i = int(round((w - crop_w)/2.))
        cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])
    else:
        cropped_image = scipy.misc.imresize(img,[resize_h, resize_w])
    if normalize:
        cropped_image = cropped_image/127.5 - 1.0
    return np.transpose(cropped_image, [2, 0, 1])
"""

# class CelebA():
#     def __init__(self):
#         datapath = os.path.join(prefix, 'celeba/aligned')
#         self.channel = 3
#         self.data = glob(os.path.join(datapath, '*.jpg'))

#     def __call__(self, batch_size, size):
#         batch_number = len(self.data)/batch_size
#         path_list = [self.data[i] for i in np.random.randint(len(self.data), size=batch_size)]
#         file_list = [p.split('/')[-1] for p in path_list]
#         batch = [get_img(img_path, True, 178, size, True) for img_path in path_list]
#         batch_imgs = np.array(batch).astype(np.float32)
#         return batch_imgs

#     def save_imgs(self, samples, file_name):
#         N_samples, channel, height, width = samples.shape
#         N_row = N_col = int(np.ceil(N_samples**0.5))
#         combined_imgs = np.ones((channel, N_row*height, N_col*width))
#         for i in range(N_row):
#             for j in range(N_col):
#                 if i*N_col+j < samples.shape[0]:
#                     combined_imgs[:,i*height:(i+1)*height, j*width:(j+1)*width] = samples[i*N_col+j]
#         combined_imgs = np.transpose(combined_imgs, [1, 2, 0])
#         scipy.misc.imsave(file_name+'.png', combined_imgs)


class CelebA():
    def __init__(self, path):
        resolution = ['data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32', 'data64x64', \
                        'data128x128', 'data256x256', 'data512x512', 'data1024x1024']
        self._base_key = 'data'
        self.dataset = h5py.File(path, 'r')
        self._len = {k:len(self.dataset[k]) for k in resolution}
        assert all([resol in self.dataset.keys() for resol in resolution])

    def __call__(self, batch_size, size, level=None):
        key = self._base_key + '{}x{}'.format(size, size)
        idx = np.random.randint(self._len[key], size=batch_size)
        batch_x = np.array([self.dataset[key][i]/127.5-1.0 for i in idx], dtype=np.float32)
        if level is not None:
            if level != int(level):
                min_lw, max_lw = int(level+1)-level, level-int(level)
                lr_key = self._base_key + '{}x{}'.format(size//2, size//2)
                low_resol_batch_x = np.array([self.dataset[lr_key][i]/127.5-1.0 for i in idx], dtype=np.float32).repeat(2, axis=2).repeat(2, axis=3)
                batch_x = batch_x * max_lw + low_resol_batch_x * min_lw
        return batch_x

    """def save_imgs(self, samples, file_name):
        N_samples, channel, height, width = samples.shape
        N_row = N_col = int(np.ceil(N_samples**0.5))
        combined_imgs = np.ones((channel, N_row*height, N_col*width))
        for i in range(N_row):
            for j in range(N_col):
                if i*N_col+j < samples.shape[0]:
                    combined_imgs[:,i*height:(i+1)*height, j*width:(j+1)*width] = samples[i*N_col+j]
        combined_imgs = np.transpose(combined_imgs, [1, 2, 0])
        scipy.misc.imsave(file_name+'.png', combined_imgs)
    """



class RandomNoiseGenerator():
    def __init__(self, size, noise_type='gaussian'):
        self.size = size
        self.noise_type = noise_type.lower()
        assert self.noise_type in ['gaussian', 'uniform']
        self.generator_map = {'gaussian': np.random.randn, 'uniform': np.random.uniform}
        if self.noise_type == 'gaussian':
            self.generator = lambda s: np.random.randn(*s)
        elif self.noise_type == 'uniform':
            self.generator = lambda s: np.random.uniform(-1, 1, size=s)

    def __call__(self, batch_size):
        return self.generator([batch_size, self.size]).astype(np.float32)


if __name__ == "__main__":
    pass
    #f = h5py.File("/Users/nduginets/PycharmProjects/PyTorch-progressive_growing_of_gans/h5.txt", 'r')
    #print(f.keys())
    #for k in f.keys():
    #    print(f[k][0])
