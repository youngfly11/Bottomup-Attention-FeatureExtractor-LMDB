import os
import io

import detectron2
import glob
import json

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
import os.path as osp
from tqdm import tqdm
# import some common libraries
import numpy as np
import cv2
import torch
# Show the image in ipynb
import PIL.Image
import pickle


# Load VG Classes
data_path = 'demo/data/1600-400-20'

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())
        
vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())


MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs
cfg = get_cfg()
cfg.merge_from_file("./configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
cfg.MODEL.WEIGHTS = "https://nlp1.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
predictor = DefaultPredictor(cfg)

# with open('/root/dataspace/Flickr30k/flickr30k_annos/train.txt', 'r') as load_f:
    # files = load_f.readlines()

# with open('/root/dataspace/Flickr30k/flickr30k_annos/test.txt', 'r') as load_f:
#     files.extend(load_f.readlines())
    
# files = [fl.split('\t')[0] for fl in files]

files = glob.glob(f'/root/dataspace/Flickr30k/flickr30k-images/*.jpg')
# files = [f'/root/dataspace/Flickr30k/flickr30k-images/{fl}.jpg' for fl in files]

det_results={0.5:{}, 0.6:{}}

for im_name in tqdm(files):
    # im = cv2.imread("./demo/data/images/input.jpg")
    im = cv2.imread(im_name)
    basename = osp.basename(im_name).split('.')[0]
    raw_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    with torch.no_grad():

        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (im_name, raw_height, raw_width))
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(im).apply_image(im)
        print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)
        
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(predictor.model.roi_heads.box2box_transform,
                                  pred_class_logits, pred_proposal_deltas,
                                  proposals, predictor.model.roi_heads.smooth_l1_beta)
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        
        # Note: BUTD uses raw RoI predictions, we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor          
        # NMS

        for nms_thresh in np.arange(0.5, 0.7, 0.1):
            instances, ids = fast_rcnn_inference_single_image(boxes, probs, image.shape[1:], score_thresh=0.1, nms_thresh=nms_thresh, topk_per_image=100)    
            instances = detector_postprocess(instances, raw_height, raw_width)
            roi_features = feature_pooled[ids].detach()
            max_attr_prob_v1 = max_attr_prob[ids].detach()
            max_attr_label_v1 = max_attr_label[ids].detach()
            instances.attr_scores = max_attr_prob_v1
            instances.attr_classes = max_attr_label_v1
            det_ret={
                "bbox": instances.pred_boxes.tensor.cpu().numpy(),
                "score": instances.scores.cpu().numpy(),
                "class": instances.pred_classes.cpu().numpy(),
                "scale": image.shape[1]/raw_height,
                "attr_score": instances.attr_scores.cpu().numpy(),
                "attr_class": instances.attr_classes.cpu().numpy()
            }
            det_results[nms_thresh][basename] = det_ret

        feature = features[0].detach().cpu().numpy()
        np.save(f"/root/dataspace/Flickr30k/flickr30k_feats/{basename}.npy", feature)

    
with open('/root/dataspace/Flickr30k/flickr30k_annos/det_score_0p1_nms_0p5.pkl', 'wb') as dump_f:
    pickle.dump(det_results[0.5], dump_f)

with open('/root/dataspace/Flickr30k/flickr30k_annos/det_score_0p1_nms_0p6.pkl', 'wb') as dump_f:
    pickle.dump(det_results[0.6], dump_f)
    

