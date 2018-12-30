"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

# 最后一层的输出　boxes object_confidence  class_confidence  共计三层
# feats 输出形状　(?,13,13,255)
# grid : 产生网格位置矩阵 ( cx,cy)
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors) # num_anchors =3

    # Reshape to batch, height, width, num_anchors, box_params.
    # reshape ->(1,1,1,3,2)    3对anchors　　[116,90,  156,198,  373,326]


    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width   (13 13)

    # K.tile 在某个维度上的复制　次数　在非1部分
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))
    # grid 产生网格位置矩阵

    #  (-1,13,13,3,85)
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    # [0 1 2 3 4 5 ....84]
    # [x y w h con class ]
    # box_xy (1,13,13,3,2)　　                            grid_shape[::-1] 反序列
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # box_wh (1,13,13,3,2) * [1,1,1,3,2]                    反序列　pw=anchors(w)/featuremap(w)
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    # box_confidence (1,13,13,3,1) 有目标为１　无目标0.5
    box_confidence = K.sigmoid(feats[..., 4:5])
    # box_class_probs (1,13,13,3,80)
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# 目标框映射到原图
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    # 颠倒数组 yx
    # _hw 在最后一个维度颠倒
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    #    input_shape (416,416)     hw
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    #     image_shape (600, 960)   hw
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    # 长宽同比例缩放    同yolo3/utils.py  letterbox_image 处理相似
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    # new_shape (260, 416)

    offset = (input_shape-new_shape)/2./input_shape

    scale = input_shape/new_shape
    # 统一缩放　再填充到固定尺寸　　左上为开始点(0,0) 图像坐标系
    box_yx = (box_yx - offset) * scale
    # 放回到原来尺寸只需要缩放　scale(
    box_hw *= scale

    # box_yx 是目标中心坐标位置
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

# feats 输出形状　(?,13,13,255)
# boxes [[xmin,ymin,xmax,ymax],  []]
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    # box_xy : 中心相对左上网格的位置　０～１
    # box_confidence : 前景和背景置信度
    # box_class_probs: 80个类别的置信度
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    # 转换为真实位置
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    # 物体置信度*类别置信度=得分
    box_scores = box_confidence * box_class_probs
    # reshape,将不同网格的值转换为框的列表。即（?,13,13,3,4）->(?,4)  ？：框的数目(后一个？)
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

            # yolo_outputs:  模型的输出【（?，13,13,255）（?，26,26,255）（?,52,52,255）】
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              # 每张图每个类别最多20个框
              max_boxes=20,
              score_threshold=.6,
              # 大于Iou阈值的被视为同一个对象　所以对于人群密度高的应该设置大值　而稀疏的应该设置小值
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    # num_layers :3
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    # 13x13 视野大配合的是　[6,7,8] 的 anchor
    # input_shape : 13x32=416
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # K.concatenate:将数据展平 ->(?,4)
    boxes = K.concatenate(boxes, axis=0)
    # box_scores = K.concatenate(box_scores, axis=0)   # ->(?,80)　每个框属于80个类别概率和为1
    box_scores = K.concatenate(box_scores, axis=0)
    # MASK掩码，过滤小于score阈值的值，只保留大于阈值的值
    mask = box_scores >= score_threshold

    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        # 从boxes筛选　类别c 得分大于阈值　的boxes
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        # 筛选出来的框的分值
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # 索引出符合条件的框及其对应的分数
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        # 输出给定位置的框
        class_boxes = K.gather(class_boxes, nms_index)
        # 输出给定位置的分置
        class_box_scores = K.gather(class_box_scores, nms_index)
        # 创建一个将所有元素都设置为1的张量
        # 这批框属于同一个类别
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
        # boses_: [[.3.],[..3.],[..4.]]  10框分为三个类别
        # scores_: [[1,2,3],[1,2,3],[1,2,3,4]] 对应分值
        # classes_:[] 对应类别
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_




# 训练部分　　成才模型输出格式的多维数组　
# 1. true box 与9个anchors 求Iou, 得到最大的anchor (373,326)
# 2. 根据anchor 找到其应该对应的　输出层　y1
# 3. 在y1层　(cx,cy)*grid_shape,找到在grid 的位置(j,i),其中(cx,cy)通过input_shape 的归一化
# 4.  (373, 326) 在　[ （116,90）,  （156,198）,  （373,326）]　索引号找到anchor 号2
# 5. 在（...,2,85） 赋予 cx,cy,w,h,k,class
#      将每张图片的boxes分配到分配到最合适的层和anchor
# 输出 y_true : [shape(m,13,13,3,85), shape(m,26,26,3,85),shape(m,52,52,3,85)]　其他地方用０填充处理
# y_true   (cx, cy, w, h) 均是相对于　input_shape!!!!
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh  N=9
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')

    # input_shape : (416,416)
    input_shape = np.array(input_shape, dtype='int32')

    # 中心坐标  宽高
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    # 中心归一化到input_shape (416,416) 0~1
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]

    # pred box-wh :e(tw)*anchors/input_shape  宽高归一化
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # m 批量
    m = true_boxes.shape[0]
    # tensor    [tensor<13,13>, tensor<26,26>, tensor<52,52>]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]

    # y_true : [shape(m,13,13,3,85), shape(m,26,26,3,85),shape(m,52,52,3,85)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    # 维度扩张  anchors shape(9,2) ----->  anchors shape(1,9,2)
    anchors = np.expand_dims(anchors, 0)

    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        # 第一张图
        wh = boxes_wh[b, valid_mask[b]]
        # 图中没有目标
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算IOU : 同一个网格有３框, 选择同true　box最大IOU的框,....以网格为中心计算的
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # Find best anchor for each true box
        # 为每个 true box 找到最大 Iou anchor box  在9个anchors 中找
        best_anchor = np.argmax(iou, axis=-1)

        # 索引号(图片框号）　　元素(373, 326)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                # if (373, 326) in [ （116,90）,  （156,198）,  （373,326）]
                if n in anchor_mask[l]:
                    # floor 向下取整
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    # k=2  (373,326) 在　[ （116,90）,  （156,198）,  （373,326）]　的为位置
                    k = anchor_mask[l].index(n)
                    # 该目标框的类别
                    c = true_boxes[b,t, 4].astype('int32')
                    # y_true[l] 代表第几层的输出
                    # b 图号　　j,i 网格坐标　　k anchors 号　0,1,2
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]  # xc,yc,w,h　(cx,cy) 似乎有问题）
                    y_true[l][b, j, i, k, 4] = 1                       # 表示存在目标
                    y_true[l][b, j, i, k, 5+c] = 1 #　类别
    # y_true(m,13,13,3,85)
    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    # 在最后一个维度上的切片
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

    # args即[*model_body.output, *y_true]
    # anchors : 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    # y_true : [shape(m,13,13,3,85), shape(m,26,26,3,85),shape(m,52,52,3,85)]
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)


    args：是一个list 组合，包括了预测值和真实值，具体如下：
    #     args[:num_layers]--预测值yolo_outputs，
    #     args[num_layers:]--真实值y_true，
    #     yolo_outputs，y_true同样是list，分别是[y1,y2,y3]三个feature map 上的的预测结果,
    #     每个y都是m*grid*grid*num_anchors*(num_classes+5),作者原文是三层，分别是(m,13,13,3,85)\
    #     (m,26,26,3,85),(m,52,52,3,85)
    #     m=batch_size

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    # 网络输入大小　model_image_size　：(416,416)
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    # [shape(13, 13), shape(26, 26), shape(52, 52)]
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    # m = batch_size
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    # 一个批次的图片在一层中的损失
    for l in range(num_layers):
        # object_mask : (m,13,13,3,1)
        object_mask = y_true[l][..., 4:5]
        # true_class_probs : (m,13,13,3,80)
        true_class_probs = y_true[l][..., 5:]

        # raw_pred : feats　　　yolo_head 解析出网路的每一层输出
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        # 在列方向的结合　[x,y,w,h]
        pred_box = K.concatenate([pred_xy, pred_wh])
        # pre_box (m,13,13,3,2) 在某层的输出

        # Darknet raw box to calculate loss.
        # y_true[0]=(m, 13, 13,3, 85)
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        # 反求　tw,th　　此处 *input_shape 是否同　preprocess_true_boxes　/ 是否多余
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        # 计算某种面积　　2-w*h (w, h,在０～１之间)
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]


        # Find ignore mask, iterate over each of batch.
        # ignore_mask(1,13,13,3,1)
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')


        def loop_body(b, ignore_mask):
            # object_mask_bool(b,13,13,3,1)--五维数组，第b张图的第l层feature map
            # 筛选出 true　box
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            # pre_box (m,13,13,3,4)  pre[b](13,13,3,4)   true_box在某层某张图的输出 : 某些位置　[2,2,0,(cx,cy,w,h)]
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            # feat_map 上与　true_boxes　的重合度小于0.5的框
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
            # 得到ignore_mask[b][13, 13, 3] 3 anchors 表示哪些预选框该忽略
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        # x,y交叉熵损失，首先要置信度不为0
        # box_loss_scale  (m,13,13,3,1)        raw_true_xy  (cx,cy)/input *grid_shap-grid 与　raw_pred[0:2]　就是tx, ty
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        #  wh_loss : 平方损失                                  raw_true_wh  log((w,h)/anchor)      raw_pred[2:4] tw, th
        # raw_pred : shape(m, 13, 13, 3, 85)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        # 置信度误差  有物体和无物体   交叉熵损失　　
        # 上：　计算标记存在物体地方的交叉熵损失, 在标记为答案的地方出错必须受罚
        # 下：　计算标记无物体出的交叉熵损失，　既然在几乎不可能的位置出现了，就该受罚
        # 只有在其周围的不会受罚
        # K.binary_crossebtropy(from_logits=True)
        # 0,1,1
        # 0.2 , 0.3 ,0.7
        # 
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        # 类别交叉熵损失，　只计算标记含有对象的地方类别损失　　惩罚力度较小
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss

#  损失是在特征图上计算的
# ignore_mask : 忽略不计算confidence 的网格
# true_box_wh ：　原图像经过平移、翻转、颜色、对比度等的变化之后的目标框并归一化（除以input_shape）, 作为　后续y_true的输入
# y_true_box_wh : 直接等同于　yolo_head 的　box_wh
# box_wh 反求在feature_map 上的值　　tw, th.
