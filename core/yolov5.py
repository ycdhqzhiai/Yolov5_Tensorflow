# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('./')
from core import utils
from core.config import cfg

import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


def mish(inputs):
    MISH_THRESH = 20.0
    tmp = inputs
    inputs = tf.where(tf.logical_and(tf.less(inputs, MISH_THRESH), tf.greater(inputs, -MISH_THRESH)),
                      tf.log(1 + tf.exp(inputs)), tf.zeros_like(inputs))
    inputs = tf.where(tf.less(inputs, -MISH_THRESH), tf.exp(inputs), inputs)
    inputs = tmp * tf.tanh(inputs)
    # return inputs * tf.tanh(tf.nn.softplus(inputs))
    return inputs


def conv(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=False, act_fun='mish'):
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = 'SAME'

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer(),
                                                 training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
            # conv = tf.concat(conv, bias)

        if activate == True:
            if act_fun == 'mish':
                conv = mish(conv)
            else:
                conv = tf.nn.leaky_relu(conv, alpha=0.1)
    return conv


def res_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):
    short_cut = input_data
    with tf.variable_scope(name):
        input_data = conv(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                          trainable=trainable, name='conv1', act_fun='mish')
        input_data = conv(input_data, filters_shape=(3, 3, filter_num1, filter_num2),
                          trainable=trainable, name='conv2', act_fun='mish')
        residual_ouput = input_data + short_cut
    return residual_ouput


def upsample(input_data, name, method='deconv'):
    assert method in ['resize', 'deconv']
    if method == 'resize':
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
    
    if method == 'deconv':
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        num_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, num_filter, kernel_size=2, padding='same',
                                            strides=(2, 2), kernel_initializer=tf.random_normal_initializer())
    return output


def csp1stage(input_data, trainable, filters, init_depth_size, stage_num):
    '''CSPNets stage
        param input_data: The input tensor
        param trainable: A bool parameter, True ==> training, False ==> not train.
        param filters: Filter nums
        param loop: ResBlock loop nums
        param layer_nums: Counter of Conv layers
        param route_nums: Counter of route nums
        param res_nums: Counter of ResBlock nums
    return: Output tensors and the last Conv layer counter of this stage'''
    with tf.variable_scope('cspstage_%d' % (stage_num)):
        c = filters
        route = input_data
        route = conv(route, (1, 1, c, int(c / 2)), trainable=trainable, name='conv_route', act_fun='mish')
        input_data = conv(input_data, (1, 1, c, int(c / 2)), trainable=trainable, name='conv', act_fun='mish')
        
        for i in range(init_depth_size):
            input_data = res_block(input_data, int(c / 2), int(c / 2), int(c / 2), trainable=trainable, name='residual%d_%d' % (i, init_depth_size))
        
        input_data = conv(input_data, (1, 1, int(c / 2), int(c / 2)), trainable=trainable, name='conv_residualout', act_fun='mish')
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(),
                                            moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer(),
                                            training=trainable)
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1)

        input_data = conv(input_data, (1, 1, int(c), int(c)), trainable=trainable, name='conv_cspout', act_fun='mish')

    return input_data

def csp2stage(input_data, trainable, filters, init_depth_size, stage_num):

    with tf.variable_scope('cspstage_%d' % stage_num):
        c = filters
        route = input_data
        route = conv(route, (1, 1, c, int(c / 2)), trainable=trainable, name='conv_route', act_fun='mish')
        input_data = conv(input_data, (1, 1, c, int(c / 2)), trainable=trainable, name='conv', act_fun='mish')

        for i in range(init_depth_size):
            input_data = conv(input_data, (1, 1, int(c / 2), int(c / 2)), trainable=trainable, name='1x1_%d' % (i), act_fun='mish')
            input_data = conv(input_data, (3, 3, int(c / 2), int(c / 2)), trainable=trainable, name='3x3_%d' % (i), act_fun='mish')


        input_data = conv(input_data, (1, 1, int(c / 2), int(c / 2)), trainable=trainable, name='conv_out', act_fun='mish')
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(),
                                            moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer(),
                                            training=trainable)
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1)
        input_data = conv(input_data, (1, 1, int(c), int(c)), trainable=trainable, name='conv_cspout', act_fun='mish')

    return input_data

def sppstage(input_data, filters_shape, trainable):
    with tf.variable_scope('sppstage'):
        input_data = conv(input_data, filters_shape, trainable=trainable, name='conv_spp', act_fun='mish')
        maxpool1 = tf.nn.max_pool(input_data, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
        maxpool2 = tf.nn.max_pool(input_data, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
        maxpool3 = tf.nn.max_pool(input_data, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
        input_data = tf.concat([maxpool1, maxpool2, maxpool3, input_data], axis=-1)
    return input_data

def cspdarknet53(input_data, trainable, init_width_size, init_depth_size):
    '''CSPDarknet53 body; source: https://arxiv.org/pdf/1911.11929.pdf
        param input_data: Input tensor
        param trainable: A bool parameter, True ==> training, False ==> not train.
    return: Three stage tensors'''
    # yolov5s sample remove slice op

    conv_num = 0
    csp_num = 0
    with tf.variable_scope('cspdarknet53'):
        # 3x608x608 -> 32x304x304
        input_data = conv(input_data, (3, 3, 3, init_width_size), trainable=trainable, name='conv%d' % (conv_num),
                        downsample=True, bn=False, act_fun='mish')
        conv_num += 1
        # 32x304x304 -> 64x152x152
        input_data = conv(input_data, (1, 1, init_width_size, 2*init_width_size), trainable=trainable, name='conv%d' % (conv_num),
                        downsample=True, act_fun='mish')
        conv_num += 1

        # 64x152x152 -> 64x152x152
        input_data = csp1stage(input_data, trainable, 2*init_width_size, init_depth_size, csp_num)
        csp_num += 1

        # 64x152x152 -> 128x76x76
        input_data = conv(input_data, (3, 3, 2*init_width_size, 4*init_width_size), trainable=trainable, 
                        name='conv%d' % (conv_num), downsample=True, act_fun='mish')
        conv_num += 1

        input_data = csp1stage(input_data, trainable, 4*init_width_size, 3*init_depth_size, csp_num)
        csp_num += 1

        route_1 = input_data # 128x76x76
  
        input_data = conv(input_data, (3, 3, 4*init_width_size, 8*init_width_size), trainable=trainable, 
                        name='conv%d' % (conv_num), downsample=True, act_fun='mish')
        conv_num += 1


        input_data = csp1stage(input_data, trainable, 8*init_width_size, 3*init_depth_size, csp_num)
        csp_num += 1

        route_2 = input_data


        input_data = conv(input_data, (3, 3, 8*init_width_size, 16*init_width_size), trainable=trainable,
                        name='conv%d' % (conv_num), downsample=True, act_fun='mish')
        conv_num += 1

        input_data = sppstage(input_data, (3, 3, 16*init_width_size, 8*init_width_size), trainable=trainable)

    
        # 4096x38x38 -> 1024x38x38
        route_3 = conv(input_data, (1, 1, 32*init_width_size, 16*init_width_size), trainable=trainable, 
                        name='conv%d' % (conv_num), act_fun='mish')
        conv_num += 1
    return route_1, route_2, route_3, conv_num, csp_num

class YOLOV5(object):
    def __init__(self, input_data, trainable):
        self.trainable = trainable
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD

        self.width_scale = cfg.YOLO.WIDTH_SCALE_V5
        self.depth_scale = cfg.YOLO.DEPTH_SCALE_V5

        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_network(input_data)
        except:
            raise NotImplementedError('Can not build up yolov5 network')

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])
        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])
        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])


    def __build_network(self, input_data):
        '''Build yolov5 body, including SPP, PAN, Yolov3/v4 Head/Neck.
           param input_data: Input tensor, return: Three stage outputs'''

        init_width_size = int(64 * self.width_scale)  # s=32
        if self.depth_scale == 0.33:
            init_depth_size = 1
        elif self.depth_scale == 0.67:
            init_depth_size = 2
        elif self.depth_scale == 1.33:
            init_depth_size = 4
        else:
            init_depth_size = 3
   
        route_1, route_2, route_3, conv_num, csp_num = cspdarknet53(input_data, self.trainable, init_width_size, init_depth_size)

        out_19 = csp2stage(route_3, self.trainable, 16*init_width_size, init_depth_size, csp_num)
        csp_num += 1

        out_19 = conv(out_19, (1, 1, 16*init_width_size, 8*init_width_size), trainable=self.trainable, 
                        name='conv_%d' % (conv_num), act_fun='mish')
        out_19_1 = out_19
        conv_num += 1

        out_38 = upsample(out_19, name='upsample0', method=self.upsample_method)

        out_38 = tf.concat([route_2, out_38], axis=-1)
        
        out_38 = csp2stage(out_38, self.trainable, 8*init_width_size, init_depth_size, csp_num)
        csp_num += 1

        out_38 = conv(out_38, (1, 1, 8*init_width_size, 4*init_width_size),  trainable=self.trainable, 
                        name='conv_%d' % (conv_num), act_fun='mish')
        out_38_1 = out_38
        conv_num += 1

        out_76 = upsample(out_38, name='upsample1', method=self.upsample_method)

        out_76 = tf.concat([route_1, out_76], axis=-1)        

        out_76 = csp2stage(out_76, self.trainable, 4*init_width_size, init_depth_size, csp_num)
        csp_num += 1

        conv_lbbox = conv(out_76, (1, 1, 4*init_width_size, 3 * (self.num_class + 5)), trainable=self.trainable,
                            name='conv_lbbox', activate=False, bn=False)
        conv_num += 1

        out_38_2 = conv(out_76, (3, 3, 4*init_width_size, 4*init_width_size), trainable=self.trainable, 
                        name='conv_%d' % (conv_num), downsample=True, act_fun='mish')
        conv_num += 1

        out_38 = tf.concat([out_38_1, out_38_2], axis=-1)

        out_38 = csp2stage(out_38, self.trainable, 8*init_width_size, init_depth_size, csp_num)
        csp_num += 1

        conv_mbbox = conv(out_38, (1, 1, 8*init_width_size, 3 * (self.num_class + 5)), trainable=self.trainable,
                            name='conv_mbbox', activate=False, bn=False)

        out_19_2 = conv(out_38, (3, 3, 8*init_width_size, 8*init_width_size), trainable=self.trainable, 
                        name='conv_%d' % (conv_num), downsample=True, act_fun='mish')
        conv_num += 1


        out_19 = tf.concat([out_19_1, out_19_2], axis=-1)

        out_19 = csp2stage(out_19, self.trainable, 16*init_width_size, init_depth_size, csp_num)
        csp_num += 1

        conv_sbbox = conv(out_19, (1, 1, 16*init_width_size, 3 * (self.num_class + 5)), trainable=self.trainable,
                        name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox


    def decode(self, conv_ouput, anchors, strides):
        '''Decode yolov5, use sigmoid decode conv_output.
            param conv_ouput: The output of yolov5 body.
            param anchors: The anchors
            param strides: Three dimensions, default [8, 16, 32]
        return: The predict of conv_output'''
        conv_shape = tf.shape(conv_ouput)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_ouput = tf.reshape(conv_ouput, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))
        conv_raw_xy = conv_ouput[:, :, :, :, 0:2]
        conv_raw_wh = conv_ouput[:, :, :, :, 2:4]
        conv_raw_conf = conv_ouput[:, :, :, :, 4:5]
        conv_raw_prob = conv_ouput[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        bbox_xy = (tf.sigmoid(conv_raw_xy) + xy_grid) * strides
        bbox_wh = (tf.sigmoid(conv_raw_wh) * anchors) * strides

        pred_xywh = tf.concat([bbox_xy, bbox_wh], axis=-1)
        pred_box_confidence = tf.sigmoid(conv_raw_conf)
        pred_box_class_prob = tf.sigmoid(conv_raw_prob)
        return tf.concat([pred_xywh, pred_box_confidence, pred_box_class_prob], axis=-1)


    def bbox_iou(self, boxes1, boxes2):
        '''Calculate bbox iou; source:
            param boxes1: Tensor, shape=(i1,...,iN, 4), xywh
            param boxes2: Tensor, shape=(j, 4), xywh
        return: Tensor, shape=(i1,...,iN, j)'''
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        
        iou = 1.0 * inter_area / union_area
        return iou


    def bbox_giou(self, boxes1, boxes2):
        '''Calculate giou loss; source: https://arxiv.org/abs/1902.09630
            param boxes1: Tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
            param boxes2: Tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        return: Tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)'''
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]), tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]), tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
        return giou


    def bbox_diou(self, boxes1, boxes2):
        '''Calculate diou; source: https://arxiv.org/pdf/1911.08287v1.pdf
            param boxes1: Tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
            param boxes2: Tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        return: Tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)'''
        boxes1_center, boxes2_center = boxes1[..., :2], boxes2[..., :2]
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]), tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]), tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        center_distance = tf.reduce_sum(tf.square(boxes1_center -boxes2_center), axis=-1)
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

        diou = iou - 1.0 * center_distance / (enclose_diagonal + 1e-7)
        return diou


    def bbox_ciou(self, boxes1, boxes2):
        '''Calculate ciou; source: https://arxiv.org/pdf/1911.08287v1.pdf
            param boxes1: Tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
            param boxes2: Tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        return: Tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)'''
        boxes1_1, boxes2_1 = boxes1, boxes2
        boxes1_center, boxes2_center = boxes1[..., :2], boxes2[..., :2]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]), tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]), tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        center_distance = tf.reduce_sum(tf.square(boxes1_center -boxes2_center), axis=-1)
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)
        diou = iou - 1.0 * center_distance / (enclose_diagonal + 1e-7)

        v = 4 / (np.pi * np.pi) * (tf.square(tf.math.atan2(boxes1_1[..., 2], boxes1_1[..., 3]) -
                                             tf.math.atan2(boxes2_1[..., 2], boxes2_1[..., 3])))
        alp = v / (1.0 - iou + v)
        ciou = diou - alp * v
        return ciou


    def focal_loss(self, y_true, y_pred, gamma=2.0, alpha=1):
        '''Compute focal loss; source:https://arxiv.org/abs/1708.02002
            param y_true: Ground truth targets, tensor of shape (?, num_boxes, num_classes).
            param y_pred: Predicted logits, tensor of shape (?, num_boxes, num_classes).
            param gamma: Exponent of the modulating factor (1 - p_t) ^ gamma.
            param alpha: Optional alpha weighting factor to balance positives vs negatives.
        return: Focal factor'''
        focal_loss = alpha * tf.pow(tf.abs(y_true - y_pred), gamma)
        return focal_loss


    def _label_smoothing(self, y_true, label_smoothing):
        '''Label smoothing. source: https://arxiv.org/pdf/1906.02629.pdf'''
        label_smoothing = tf.constant(label_smoothing, dtype=tf.float32)
        #return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        
        label_smoothing_num = 2
        if self.num_class > 2:
            label_smoothing_num = self.num_class

        new_y_true = y_true * (1.0 - label_smoothing) + label_smoothing / label_smoothing_num
        return new_y_true       


    def yolov5_loss(self, conv, pred, label, bboxes, stride, iou_use=1, focal_use=False, label_smoothing=0):
        '''Reture yolov5_loss tensor.
            param conv: The outputs of yolov5 body, conv_sbbox, conv_mbbox, conv_lbbox
            param pred: The outputs of decode, pred_sbbox, pred_mbbox, pred_lbbox
            param label: The input label boxes
            param bboxes: The input boxes
            param stride: Num of [8, 16, 32]
            param iou_use: The iou loss (0, 1, 2) ==> (giou, diou, ciou)
            param focal_use: The focal loss  (0, 1, 2) ==> (normal, sigmoid_focal, focal)
            param label_smoothing: The label smoothing
        return: Tensor, shape=(1,)'''

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, self.anchor_per_scale, 5 + self.num_class))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]
        if label_smoothing:
            label_prob = self._label_smoothing(label_prob, label_smoothing)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        respond_backgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        input_size = tf.cast(input_size, tf.float32)
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)

        if iou_use == 1:
            diou = tf.expand_dims(self.bbox_diou(pred_xywh, label_xywh), axis=-1)
            iou_loss = respond_bbox * bbox_loss_scale * (1 - diou)
        elif iou_use == 2:
            ciou = tf.expand_dims(self.bbox_ciou(pred_xywh, label_xywh), axis=-1)
            iou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)
        else:
            giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
            iou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        if focal_use:
            focal = self.focal_loss(respond_bbox, pred_conf)
            conf_loss = focal * (respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf) + \
                                 respond_backgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf))
            class_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        else:
            conf_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf) + \
                        respond_backgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            class_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3, 4]))
        return iou_loss, conf_loss, class_loss


    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox, iou_use, focal_use, label_smoothing):
        '''Compute loss; location loss, confidence loss, class prob loss '''
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.yolov5_loss(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox, stride=self.strides[0],
                                          iou_use=iou_use, focal_use=focal_use, label_smoothing=label_smoothing)        
        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.yolov5_loss(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox, stride=self.strides[1], 
                                          iou_use=iou_use, focal_use=focal_use, label_smoothing=label_smoothing)
        with tf.name_scope('lager_box_loss'):
            loss_lbbox = self.yolov5_loss(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox, stride=self.strides[2], 
                                          iou_use=iou_use, focal_use=focal_use, label_smoothing=label_smoothing)

        with tf.name_scope('iou_loss'):
            iou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]
        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]
        with tf.name_scope('class_loss'):
            class_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]
        return iou_loss, conf_loss, class_loss


if __name__ == "__main__":
    input_data = tf.placeholder(dtype=tf.float32, shape=(1,608,608,3), name='input_data')
    trainable = tf.placeholder(dtype=tf.bool, name='training')
    model = YOLOV5(input_data, trainable)
    y19_output, y38_output, y76_output = model.conv_lbbox, model.conv_mbbox, model.conv_sbbox
    print(y19_output, y38_output, y76_output)
    print(model.pred_lbbox, model.pred_mbbox, model.pred_sbbox)