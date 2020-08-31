'''
Author: your name
Date: 2020-08-25 11:41:16
LastEditTime: 2020-08-25 13:52:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \\Yolov5_tf\\core\\config.py
'''
#! /usr/bin/env python
# -*- coding: utf-8 -*-
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()


# Set the class name
__C.YOLO.NET_TYPE = 'darknet53' # 'darknet53' 'mobilenetv2'
__C.YOLO.CLASSES = 'data/classes/mnist.name'
__C.YOLO.ANCHORS = 'data/anchors/basline_anchors.txt' # yolov3/5 : yolo_anchors.txt; yolov4 : yolov4_anchors.txt
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.STRIDES_TINY = [16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
__C.YOLO.UPSAMPLE_METHOD = 'deconv'

__C.YOLO.WIDTH_SCALE_V5 = 0.50 # yolov5 small:0.50 / middle:0.75 / large:1.00 / extend:1.25
__C.YOLO.DEPTH_SCALE_V5 = 0.33 # yolov5 small:0.33(1/3) / middle:0.67(2/3) / large:1.00 / extend:1.33(4/3)

__C.YOLO.ORIGINAL_WEIGHT = 'checkpoint/yolov3_coco.ckpt'
__C.YOLO.DEMO_WEIGHT = 'checkpoint/yolov3_coco_demo.ckpt'


# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = './data/dataset/mnist_train.txt'
__C.TRAIN.BATCH_SIZE = 8 if __C.YOLO.NET_TYPE == 'mobilenetv2' else 2
__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.LEARN_RATE_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 10
__C.TRAIN.FISRT_STAGE_EPOCHS = 100
__C.TRAIN.SECOND_STAGE_EPOCHS = 1000
__C.TRAIN.INITIAL_WEIGHT = 'ckpts/yolov3_test-loss=10.0817.ckpt-125'
__C.TRAIN.CKPT_PATH = 'ckpts'


# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = './data/dataset/mnist_test.txt'
__C.TEST.BATCH_SIZE = 2
__C.TEST.INPUT_SIZE = 416
__C.TEST.DATA_AUG = False
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = 'data/detection/'
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE = 'cpkts/yolov3_test_loss=9.2099.ckpt-5'
__C.TEST.SHOW_LABEL = True
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45
