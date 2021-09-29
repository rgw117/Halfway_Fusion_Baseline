from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from data.KAIST_dataset import KAISTPed
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt

# IMAGE_MEAN = (0.3465,  0.3219,  0.2842)
# IMAGE_STD = (0.2358, 0.2265, 0.2274)

# LWIR_MEAN = (0.1598)
# LWIR_STD = (0.0813)

# # def inverse_normalize(img):
# #     if opt.caffe_pretrain:
# #         img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
# #         return img[::-1, :, :]
# #     # approximate un-normalize for visualize
# #     return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(vis_img, lwir_img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize_vis = tvtsf.Normalize(mean=[0.3465, 0.3219, 0.2842],std=[0.2358, 0.2265, 0.2274])                  
    vis_img = normalize_vis(t.from_numpy(vis_img).float())

    normalize_lwir = tvtsf.Normalize(mean=[0.3465],std=[0.2358])                  
    lwir_img = normalize_lwir(t.from_numpy(lwir_img).float())

    return vis_img.numpy(), lwir_img.numpy()


def preprocess(vis_img, lwir_img):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """

    vis_img = vis_img / 255.
    lwir_img = lwir_img / 255.

    normalize = pytorch_normalze(vis_img,lwir_img)

    return normalize 


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        vis_img, lwir_img, bbox, label = in_data
        _, H, W = lwir_img.shape
        vis_img,lwir_img = preprocess(vis_img, lwir_img)
        scale = 1

        # horizontally flip
        vis_img,lwir_img, params = util.random_flip(
            vis_img, lwir_img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (H, W), x_flip=params['x_flip'])

        return vis_img, lwir_img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db =  KAISTPed('train-all-02.txt')
#         self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx): 
        
#         ori_img, bbox, label, difficult = self.db.get_example(idx)
        vis_ori_img, lwir_ori_img, bbox, label, difficult = self.db.pull_item(idx)
        vis_ori_img = np.array(vis_ori_img).reshape(3,512,640)
        lwir_ori_img = np.array(lwir_ori_img).reshape(1,512,640)
        vis_img, lwir_img, bbox, label, scale = self.tsf((vis_ori_img, lwir_ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.

        return vis_img.copy(),lwir_img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        # self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)
        self.db =  KAISTPed('test-all-20.txt')


    def __getitem__(self, idx):
        # ori_img, bbox, label, difficult = self.db.get_example(idx)
        vis_ori_test_img, lwir_ori_test_img, bbox, label, difficult = self.db.pull_item(idx)
        vis_ori_test_img = np.array(vis_ori_test_img).reshape(3,512,640)
        lwir_ori_test_img = np.array(lwir_ori_test_img).reshape(1,512,640)
        vis_ori_test_img,lwir_ori_test_img = preprocess(vis_ori_test_img, lwir_ori_test_img)

        return vis_ori_test_img, lwir_ori_test_img, vis_ori_test_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
