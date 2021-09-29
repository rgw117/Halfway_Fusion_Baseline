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
JSON_GT_FILE = os.path.join( DB_ROOT, 'kaist_annotations_test20.json' )
cocoGt = COCO(JSON_GT_FILE)

load_path = './checkpoints/2019-09-05_00h24m_train_base_halfway_0905_15annotation/fasterrcnn_4'
save_path = './checkpoints/2019-09-05_00h24m_train_base_halfway_0905_15annotation'
epoch = 66

################################################################################

# random seed fix 
torch.manual_seed(12)
torch.cuda.manual_seed(12)
torch.backends.cudnn.deterministic=True

################################################################################
def eval(dataloader, faster_rcnn, jobs_dir=None, epoch=None):

    fig_test,  ax_test  = plt.subplots(figsize=(18,15))

    print('\nstart_evaluation\n')   
    
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()

    result_test = []
    
    for ii, (vis_test_image, lwir_test_iamge, sizes, _, _, _) in tqdm(enumerate(dataloader), desc='Evaluating'):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(vis_test_image, lwir_test_iamge, [sizes])
        
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

    # rstFile = os.path.join('./checkpoints/2019-08-13_17h02m_train_2015/COCO_TEST_det_0.json')
        
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


def evaluation(**kwargs):

    opt._parse(kwargs)

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

    trainer.load(load_path)
    print('load pretrained model from %s' % load_path)
        
    eval(test_dataloader, faster_rcnn, jobs_dir=save_path, epoch= epoch)

    lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    
    evaluation()