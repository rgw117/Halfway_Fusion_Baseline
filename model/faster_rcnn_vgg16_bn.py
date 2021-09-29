from __future__ import  absolute_import
import torchvision
import torch as t
from torch import nn
from utils import *
from itertools import product as product
import torch.nn.functional as F
from torchvision.models import vgg16, vgg16_bn
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt

class VGG_vis_lwir_feature(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(VGG_vis_lwir_feature, self).__init__()

          ##################################################################################
        # Standard convolutional layers in VGG16 for vis
        self.conv1_1_vis = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_1_bn_vis = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_vis = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1_2_bn_vis = nn.BatchNorm2d(64, affine=True)

        self.pool1_vis = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv2_1_vis = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_1_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_vis = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2_2_bn_vis = nn.BatchNorm2d(128, affine=True)
        
        self.pool2_vis = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv3_1_vis = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_1_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_2_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3_bn_vis = nn.BatchNorm2d(256, affine=True)

        self.pool3_vis = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv4_1_vis = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_1_bn_vis = nn.BatchNorm2d(512, affine=True)
        self.conv4_2_vis = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_2_bn_vis = nn.BatchNorm2d(512, affine=True)
        self.conv4_3_vis = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_bn_vis = nn.BatchNorm2d(512, affine=True)

        ##################################################################################
        # Standard convolutional layers in VGG16 for lwir
        self.conv1_1_lwir = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True)  # stride = 1, by default
        self.conv1_1_bn_lwir = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_lwir = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1_2_bn_lwir = nn.BatchNorm2d(64, affine=True)
        
        self.pool1_lwir = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv2_1_lwir = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_1_bn_lwir = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_lwir = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2_2_bn_lwir = nn.BatchNorm2d(128, affine=True)
        
        self.pool2_lwir = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv3_1_lwir = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_1_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_2_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3_bn_lwir = nn.BatchNorm2d(256, affine=True)

        self.pool3_lwir = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv4_1_lwir = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_1_bn_lwir = nn.BatchNorm2d(512, affine=True)
        self.conv4_2_lwir = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_2_bn_lwir = nn.BatchNorm2d(512, affine=True)
        self.conv4_3_lwir = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_bn_lwir = nn.BatchNorm2d(512, affine=True)

        ##################################################################################

        # NIN convolutional layer for Dimensionality Reduction
        self.conv1x1 = nn.Conv2d(1024,512,kernel_size=1, padding=0, stride=1, bias=False)
        self.conv1x1_bn = nn.BatchNorm2d(512, affine=True)
        
        nn.init.normal_(self.conv1x1.weight, mean=0, std=0.01)
        ##################################################################################
        # Fusion layer
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512, affine=True)

        # Load pretrained layers
        self.load_pretrained_layers()
    
    def forward(self, vis_image, lwir_image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 512, 640)
        :return: lower-level feature maps conv4_3 and conv7
        """
        ################################################################################## vis

        out_vis = F.relu(self.conv1_1_bn_vis(self.conv1_1_vis(vis_image.cuda())))  # (N, 64, 512, 640)
        out_vis = F.relu(self.conv1_2_bn_vis(self.conv1_2_vis(out_vis)))  # (N, 64, 512, 640)
        
        out_vis = self.pool1_vis(out_vis)  # (N, 64, 256, 320)

        out_vis = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(out_vis)))  # (N, 128, 256, 320)
        out_vis = F.relu(self.conv2_2_bn_vis(self.conv2_2_vis(out_vis)))  # (N, 128, 256, 320)
        
        out_vis = self.pool2_vis(out_vis)  # (N, 128, 128, 160)

        out_vis = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis)))  # (N, 256, 128, 160)
        out_vis = F.relu(self.conv3_2_bn_vis(self.conv3_2_vis(out_vis)))  # (N, 256, 128, 160)
        out_vis = F.relu(self.conv3_3_bn_vis(self.conv3_3_vis(out_vis)))  # (N, 256, 128, 160)
        
        out_vis = self.pool3_vis(out_vis)  # (N, 256, 64, 80)

        out_vis = F.relu(self.conv4_1_bn_vis(self.conv4_1_vis(out_vis)))  # (N, 512, 64, 80)
        out_vis = F.relu(self.conv4_2_bn_vis(self.conv4_2_vis(out_vis)))  # (N, 512, 64, 80)
        out_vis = F.relu(self.conv4_3_bn_vis(self.conv4_3_vis(out_vis)))  # (N, 512, 64, 80)
        
        # out_vis = self.pool4_vis(out_vis) # (N, 512, 32, 40)

        ################################################################################## lwir

        out_lwir = F.relu(self.conv1_1_lwir(lwir_image))  # (N, 64, 512, 640)
        out_lwir = F.relu(self.conv1_2_lwir(out_lwir))  # (N, 64, 512, 640)
        out_lwir = self.pool1_lwir(out_lwir)  # (N, 64, 256, 320)

        out_lwir = F.relu(self.conv2_1_lwir(out_lwir))  # (N, 128, 256, 320)
        out_lwir = F.relu(self.conv2_2_lwir(out_lwir))  # (N, 128, 256, 320)
        out_lwir = self.pool2_lwir(out_lwir)  # (N, 128, 128, 160)

        out_lwir = F.relu(self.conv3_1_lwir(out_lwir))  # (N, 256, 128, 160)
        out_lwir = F.relu(self.conv3_2_lwir(out_lwir))  # (N, 256, 128, 160)
        out_lwir = F.relu(self.conv3_3_lwir(out_lwir))  # (N, 256, 128, 160)
        out_lwir = self.pool3_lwir(out_lwir)  # (N, 256, 64, 80)

        out_lwir = F.relu(self.conv4_1_lwir(out_lwir))  # (N, 512, 64, 80)
        out_lwir = F.relu(self.conv4_2_lwir(out_lwir))  # (N, 512, 64, 80)
        out_lwir = F.relu(self.conv4_3_lwir(out_lwir))  # (N, 512, 64, 80)
        # out_lwir = self.pool4_lwir(out_lwir) # (N, 512, 32, 40)

       ##################################################################################

        out_lwir = F.relu(self.conv1_1_bn_lwir(self.conv1_1_lwir(lwir_image)))  # (N, 64, 512, 640)
        out_lwir = F.relu(self.conv1_2_bn_lwir(self.conv1_2_lwir(out_lwir)))  # (N, 64, 512, 640)
        
        out_lwir = self.pool1_lwir(out_lwir)  # (N, 64, 256, 320)

        out_lwir = F.relu(self.conv2_1_bn_lwir(self.conv2_1_lwir(out_lwir)))  # (N, 128, 256, 320)
        out_lwir = F.relu(self.conv2_2_bn_lwir(self.conv2_2_lwir(out_lwir)))  # (N, 128, 256, 320)
        
        out_lwir = self.pool2_lwir(out_lwir)  # (N, 128, 128, 160)

        out_lwir = F.relu(self.conv3_1_bn_lwir(self.conv3_1_lwir(out_lwir)))  # (N, 256, 128, 160)
        out_lwir = F.relu(self.conv3_2_bn_lwir(self.conv3_2_lwir(out_lwir)))  # (N, 256, 128, 160)
        out_lwir = F.relu(self.conv3_3_bn_lwir(self.conv3_3_lwir(out_lwir)))  # (N, 256, 128, 160)
        
        out_lwir = self.pool3_lwir(out_lwir)  # (N, 256, 64, 80)

        out_lwir = F.relu(self.conv4_1_bn_lwir(self.conv4_1_lwir(out_lwir)))  # (N, 512, 64, 80)
        out_lwir = F.relu(self.conv4_2_bn_lwir(self.conv4_2_lwir(out_lwir)))  # (N, 512, 64, 80)
        out_lwir = F.relu(self.conv4_3_bn_lwir(self.conv4_3_lwir(out_lwir)))  # (N, 512, 64, 80)
        
        # out_lwir = self.pool4_lwir(out_lwir) # (N, 512, 32, 40)

        ##################################################################################

        concatenate_feature = t.cat((out_vis,out_lwir), 1) # (N, 1024, 32, 40)

        # Network In Network
        # concatenate_feature = F.avg_pool2d(concatenate_feature, kernel_size=3, stride=1, padding=1) # (N, 1024, 32, 40)
        concatenate_feature = F.relu(self.conv1x1_bn(self.conv1x1(concatenate_feature))) # (N, 512, 32, 40)

        concatenate_feature = F.relu(self.conv5_1_bn(self.conv5_1(concatenate_feature)))  # (N, 512, 32, 40)
        concatenate_feature = F.relu(self.conv5_2_bn(self.conv5_2(concatenate_feature)))  # (N, 512, 32, 40)
        concatenate_feature = F.relu(self.conv5_3_bn(self.conv5_3(concatenate_feature)))  # (N, 512, 32, 40)

        # out_lwir = F.relu(self.conv5_1(out_lwir))  # (N, 512, 32, 40)
        # out_lwir = F.relu(self.conv5_2(out_lwir))  # (N, 512, 32, 40)
        # out_lwir = F.relu(self.conv5_3(out_lwir))  # (N, 512, 32, 40)
        # out_lwir = self.pool5(out_lwir)  # (N, 512, 16, 20)
        
        return concatenate_feature
 

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """

        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = vgg16_bn(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
    
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:70]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[70:-28]):  # excluding conv6 and conv7 parameters            
            if param == 'conv1_1_lwir.weight':
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]][:, 0:1, :, :]              
            else:
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[-21:]):  # excluding conv6 and conv7 parameters            
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i+70]]

        self.load_state_dict(state_dict)

        # # freeze top4 conv (vis)
        self.conv1_1_vis.requires_grad = False
        self.conv1_1_bn_vis.requires_grad = False
        self.conv1_2_vis.requires_grad = False
        self.conv1_2_bn_vis.requires_grad = False
        self.conv2_1_vis.requires_grad = False
        self.conv2_1_bn_vis.requires_grad = False
        self.conv2_2_vis.requires_grad = False
        self.conv2_2_bn_vis.requires_grad = False
        # self.conv3_1_vis.requires_grad = False
        # self.conv3_2_vis.requires_grad = False
        # self.conv3_3_vis.requires_grad = False
        # self.conv4_1_vis.requires_grad = False
        # self.conv4_2_vis.requires_grad = False
        # self.conv4_3_vis.requires_grad = False

        # # freeze top4 conv (lwir)
        self.conv1_1_lwir.requires_grad = False
        self.conv1_1_bn_lwir.requires_grad = False
        self.conv1_2_lwir.requires_grad = False
        self.conv1_2_bn_lwir.requires_grad = False
        self.conv2_1_lwir.requires_grad = False
        self.conv2_1_bn_lwir.requires_grad = False
        self.conv2_2_lwir.requires_grad = False
        self.conv2_2_bn_lwir.requires_grad = False
        # self.conv3_1_lwir.requires_grad = False
        # self.conv3_2_lwir.requires_grad = False
        # self.conv3_3_lwir.requires_grad = False
        # self.conv4_1_lwir.requires_grad = False
        # self.conv4_2_lwir.requires_grad = False
        # self.conv4_3_lwir.requires_grad = False

        print("\nLoaded base vis&lwir_fusion_model.\n")


class VGG_vis_lwir_classifier(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(VGG_vis_lwir_classifier, self).__init__()

        self.Linear1 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.Linear2 = nn.Linear(in_features=4096, out_features=4096, bias=True)

    def forward(self, feature):
        """
        Forward propagation.
        input : total_feature
        output : classifier
        """

        feature = feature.view(feature.size(0), -1) 
        out = F.relu(self.Linear1(feature),inplace=True)   # (N, 512, 16, 20) 
        # out = F.dropout(out,p=0.5)
        out = F.relu(self.Linear2(out),inplace=True)
        # classifier = F.dropout(out,p=0.5)

        return out

class FasterRCNNVGG16_bn(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 8  # downsample 16x for output of conv5 in vgg16
    
    def __init__( self, n_fg_class=1, ratios=[0.41,1], anchor_scales=[2,2.52,3.18,4,5,6.36,7,8,8.82,9,10,11.2,14.5] ):
                 
        extractor = VGG_vis_lwir_feature()
        rpn = RegionProposalNetwork(512, 512,ratios=ratios,anchor_scales=anchor_scales,feat_stride=self.feat_stride)
        head = VGG16RoIHead(n_class=n_fg_class + 1,roi_size=7,spatial_scale=(1. / self.feat_stride)) # roi_size = feature map size

        super(FasterRCNNVGG16_bn, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        
        self.classifier = VGG_vis_lwir_classifier()
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size,self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """

        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
