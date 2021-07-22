import unittest
import torch
from collections import OrderedDict
from evaluate.coco_eval import run_eval
from lib.network.rtpose_vgg import get_model, use_vgg
from lib.network.openpose import OpenPose_Model, use_vgg
from torch import load

#Notice, if you using the 
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    model = get_model(trunk='vgg19')
    #model = openpose = OpenPose_Model(l2_stages=4, l1_stages=2, paf_out_channels=38, heat_out_channels=19)
    #model = torch.nn.DataParallel(model).cuda()
    w1= torch.load('pose_model.pth',map_location='cpu')
    from collections import OrderedDict

    model.load_state_dict(w1)
    model.eval()
    model.float()
   # model = model.cuda()
    
    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in 
    # this repo used 'vgg' preprocess
    run_eval(image_dir= 'coco/images/val2017', anno_file = 'coco/annotations/person_keypoints_val2017.json', vis_dir = 'coco/images/vis_val2017', model=model, preprocess='rtpose')
"""
    w2= OrderedDict()
    for k, v in w1.items():
        name=k[7:]
        w2[name]=v
"""
