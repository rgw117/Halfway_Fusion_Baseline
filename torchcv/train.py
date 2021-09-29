from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os
import pdb
import ipdb
# import matplotlib
import tqdm
print(tqdm.__version__)
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667

################################################################################

### Evaluation
import os
import os.path
from torchcv.evaluations.coco import COCO
from torchcv.evaluations.eval_MR_multisetup import COCOeval
import torch
import json
import matplotlib.pyplot as plt
from torchcv.utils import Timer, kaist_results_file as write_result, write_coco_format as write_result_coco

annType = 'bbox'

DB_ROOT = './datasets/kaist-rgbt'
JSON_GT_FILE = os.path.join( DB_ROOT, 'kaist_annotations_test20_2015.json' )
cocoGt = COCO(JSON_GT_FILE)

# Parameters
checkpoint_root = './checkpoints'
data_folder = './datasets/kaist-rgbt/'
input_size = [512., 640.]

# Load data
# load_path = './checkpoints/2019-08-09_15h45m_train_lwir/fasterrcnn_6'
load_path = None

################################################################################
### tensorboardX & log
import logging
import logging.handlers
from datetime import datetime
from tensorboardX import SummaryWriter
from torchcv.utils import run_tensorboard
port = 8817

################################################################################

# random seed fix 
torch.manual_seed(12)
torch.cuda.manual_seed(12)
torch.backends.cudnn.deterministic=True

################################################################################
def eval(dataloader, faster_rcnn, epoch, jobs_dir):

    fig_test,  ax_test  = plt.subplots(figsize=(18,15))

    print('\nstart_evaluation\n')
    
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()

    result_test = []
    
    for ii, (vis_test_image, lwir_test_iamge, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader), desc='Evaluating'):
        
        # if ii == 50 : break

        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(vis_test_image, lwir_test_iamge, [sizes])
        # import pdb
        # pdb.set_trace()

        # gt_bboxes += list(gt_bboxes_.numpy())
        # gt_labels += list(gt_labels_.numpy())
        # gt_difficults += list(gt_difficults_.numpy())
        
        # pred_bboxes += pred_bboxes_
        # pred_labels += pred_labels_
        # pred_scores += pred_scores_

        for box, label, score in zip(pred_bboxes_[0],pred_labels_[0],pred_scores_[0]) :

            bb = box.tolist()
            result_test.append( {\
                            'image_id': ii, \
                            'category_id': label.item()+1, \
                            'bbox': [bb[1],bb[0],bb[3]-bb[1],bb[2]-bb[0]], \
                            'score': score.item() } )


    print('\nCOCO_Missrate_test\n')

    rstFile = os.path.join(jobs_dir, 'COCO_TEST_det_{:d}.json'.format(epoch))            
    write_result_coco(result_test, rstFile)
        
    try:

        cocoDt = cocoGt.loadRes(rstFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.catIds  = [1]    
        cocoEval.evaluate(0)
        cocoEval.accumulate()
        curPerf = cocoEval.summarize(0)    

        cocoEval.draw_figure(ax_test, rstFile.replace('json', 'jpg'))                
        #writer.add_scalars('LAMR/fppi', {'test': curPerf}, epoch)

        print('Recall: {:}'.format( 1-cocoEval.eval['yy'][0][-1] ) )

    except:
        import torchcv.utils.trace_error
        print('[Error] cannot evaluate by cocoEval. ')




def train(**kwargs):

    opt._parse(kwargs)

    ##########################################################################################################################################################################
    print('Setting..')

    exp_time  = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    exp_name = ('_' + opt.exp_name)
    jobs_dir = os.path.join( 'checkpoints', exp_time + opt.exp_name )

    snapshot_dir    = os.path.join( jobs_dir, 'snapshots' )
    tensorboard_dir    = os.path.join( jobs_dir, 'tensorboardX' )
    if not os.path.exists(snapshot_dir):        os.makedirs(snapshot_dir)
    if not os.path.exists(tensorboard_dir):     os.makedirs(tensorboard_dir)
    run_tensorboard( tensorboard_dir, port )

    ### Backup current source codes
    
    import tarfile
    tar = tarfile.open( os.path.join(jobs_dir, 'sources.tar'), 'w' )
    tar.add( 'torchcv' )
    tar.add( 'data' )
    tar.add( 'docker')
    tar.add('misc')
    tar.add('model')
    tar.add('utils')    
    tar.add( __file__ )

    import glob
    for file in sorted( glob.glob('*.py') ):
        tar.add( file )

    tar.close()

    ### Set logger
    
    writer = SummaryWriter(os.path.join(jobs_dir, 'tensorboardX'))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(levelname)s] [%(asctime)-11s] %(message)s')
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    logger.addHandler(h)

    h = logging.FileHandler(os.path.join(jobs_dir, 'log_{:s}.txt'.format(exp_time)))
    h.setFormatter(fmt)
    logger.addHandler(h)

    logger.info('Exp time: {}'.format(exp_time))

    #########################################################################################################################################################################
    print('load data')
    
    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                #   pin_memory=True, \
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')

    #★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    #★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    
    pass_count = 0
    plot_count = 0
    best_miss_rate = 100
    lr_ = opt.lr

    if load_path is not None :
        trainer.load(load_path)
        print('load pretrained model from %s' % load_path)


    for epoch in range(opt.epoch):
        trainer.reset_meters()

        forward_count = 0
        pass_count = 0

        for ii, (vis_image, lwir_iamge, bbox_, label_, scale) in tqdm(enumerate(dataloader), desc='Training'):
        
            # if ii == 50 : break

            try :
                if len(bbox_[0]) == 1 :
                    pass_count += 1
                    continue
                else :
                    bbox_ = bbox_[label_==1][None,:,:]
                    label_ = label_[label_==1][None,:] -1

                    if len(bbox_[0]) < 1 :
                        pass_count += 1
                        continue
                    
                    forward_count += 1

            except :
                import pdb
                pdb.set_trace()
            
            scale = at.scalar(scale)
            vis_image, lwir_iamge, bbox, label = vis_image.cuda().float(), lwir_iamge.cuda().float(), bbox_.cuda(), label_.cuda()          
            trainer.train_step(vis_image, lwir_iamge, bbox, label, scale)

            if (forward_count + 1) % opt.plot_every == 0:

                logger.info('\nEpoch : {}\n'
                            'Batch : {}/{}\n'
                            'Loss : \nrpn_loc : {} \t rpn_cls : {} \n roi_loc : {} \t roi_cls : {} \n total : {}'.format(epoch,ii,len(dataloader),\
                            trainer.get_meter_data()['rpn_loc_loss'],trainer.get_meter_data()['rpn_cls_loss'],trainer.get_meter_data()['roi_loc_loss'],\
                            trainer.get_meter_data()['roi_cls_loss'],trainer.get_meter_data()['total_loss']))
                
                writer.add_scalars('train/loss', {'rpn_loc': trainer.get_meter_data()['rpn_loc_loss'] },global_step=epoch*len(dataloader)+ii)
                writer.add_scalars('train/loss', {'rpn_cls': trainer.get_meter_data()['rpn_cls_loss'] },global_step=epoch*len(dataloader)+ii)
                writer.add_scalars('train/loss', {'roi_loc': trainer.get_meter_data()['roi_loc_loss'] },global_step=epoch*len(dataloader)+ii)
                writer.add_scalars('train/loss', {'roi_cls': trainer.get_meter_data()['roi_cls_loss'] },global_step=epoch*len(dataloader)+ii)
                writer.add_scalars('train/loss', {'total': trainer.get_meter_data()['total_loss'] },global_step=epoch*len(dataloader)+ii)

        print('\nPass Count  : {:d}   /// Training Image : {:d}\n'.format(pass_count,(len(dataloader)-pass_count)))

        eval(test_dataloader, faster_rcnn, epoch, jobs_dir)

        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']

        save_path = trainer.save(epoch=epoch,save_path=jobs_dir)
        
        # if miss_rate < best_miss_rate:
        #     best_miss_rate = miss_rate
        #     best_path = trainer.save(epoch=epoch,save_path=jobs_dir)

        if epoch % 10 == 0:
            trainer.load(save_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        # if epoch == 6: 
        #     break


if __name__ == '__main__':
    import fire

    fire.Fire()
