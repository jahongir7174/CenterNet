import copy
import math
import random

import cv2
import numpy
import torch
from torch.nn.functional import max_pool2d


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def load_image(self, index):
    image = cv2.imread(self.filenames[index])
    h, w = image.shape[:2]
    r = self.input_size / max(h, w)
    if r != 1:
        resample = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
        image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=resample)
    return image, (h, w), image.shape[:2]


def load_mosaic(self, index):
    label4 = []
    image4 = numpy.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=numpy.uint8)
    y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = (None, None, None, None, None, None, None, None)

    xc = int(random.uniform(self.input_size // 2, 2 * self.input_size - self.input_size // 2))
    yc = int(random.uniform(self.input_size // 2, 2 * self.input_size - self.input_size // 2))

    indices = [index] + random.choices(self.indices, k=3)
    random.shuffle(indices)

    for i, index in enumerate(indices):
        # Load image
        image, _, (h, w) = load_image(self, index)
        if i == 0:  # top left
            x1a = max(xc - w, 0)
            y1a = max(yc - h, 0)
            x2a = xc
            y2a = yc
            x1b = w - (x2a - x1a)
            y1b = h - (y2a - y1a)
            x2b = w
            y2b = h
        elif i == 1:  # top right
            x1a = xc
            y1a = max(yc - h, 0)
            x2a = min(xc + w, self.input_size * 2)
            y2a = yc
            x1b = 0
            y1b = h - (y2a - y1a)
            x2b = min(w, x2a - x1a)
            y2b = h
        elif i == 2:  # bottom left
            x1a = max(xc - w, 0)
            y1a = yc
            x2a = xc
            y2a = min(self.input_size * 2, yc + h)
            x1b = w - (x2a - x1a)
            y1b = 0
            x2b = w
            y2b = min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a = xc
            y1a = yc
            x2a = min(xc + w, self.input_size * 2)
            y2a = min(self.input_size * 2, yc + h)
            x1b = 0
            y1b = 0
            x2b = min(w, x2a - x1a)
            y2b = min(y2a - y1a, h)

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        # Labels
        label = self.labels[index].copy()
        if len(label):
            label[:, 1:] = wh2xy(label[:, 1:], w, h, pad_w, pad_h)
        label4.append(label)

    # Concat/clip labels
    label4 = numpy.concatenate(label4, 0)
    for x in label4[:, 1:]:
        numpy.clip(x, 0, 2 * self.input_size, out=x)

    # Augment
    image4, label4 = random_perspective(image4, label4, self.input_size)

    return image4, label4


def augment_hsv(image):
    # HSV color-space augmentation
    r = numpy.random.uniform(-1, 1, 3) * [.015, .7, .4] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def resize(image, input_size):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(1.0, input_size / shape[0], input_size / shape[1])

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)


def box_candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)  # candidates


def random_perspective(samples, targets, input_size):
    # Center
    center = numpy.eye(3)
    center[0, 2] = -float(input_size)  # x translation (pixels)
    center[1, 2] = -float(input_size)  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)
    perspective[2, 0] = random.uniform(-0.0, 0.0)  # x perspective (about y)
    perspective[2, 1] = random.uniform(-0.0, 0.0)  # y perspective (about x)

    # Rotation and Scale
    rotation = numpy.eye(3)
    a = random.uniform(-0, 0)
    s = random.uniform(1 - 0.5, 1 + 0.5)
    rotation[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-0.0, 0.0) * math.pi / 180)  # x shear (deg)
    shear[1, 0] = math.tan(random.uniform(-0.0, 0.0) * math.pi / 180)  # y shear (deg)

    # Translation
    translation = numpy.eye(3)
    translation[0, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * input_size  # x translation (pixels)
    translation[1, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * input_size  # y translation (pixels)

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translation @ shear @ rotation @ perspective @ center
    # image changed
    samples = cv2.warpAffine(samples, matrix[:2], dsize=(input_size, input_size))

    n = len(targets)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, input_size)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, input_size)

        # filter candidates
        i = box_candidates(targets[:, 1:5].T * s, new.T)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return samples, targets


def mix_up(image1, label1, image2, label2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = numpy.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image1 * r + image2 * (1 - r)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


def clip(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale(coords, shape1, shape2, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])  # gain  = old / new
        pad = (shape1[1] - shape2[1] * gain) / 2, (shape1[0] - shape2[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip(coords, shape2)
    return coords


def xy2wh(x, w=640, h=640):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    clip(x, (h - 1E-3, w - 1E-3))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    m_rec = numpy.concatenate(([0.0], recall, [1.0]))
    m_pre = numpy.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate
    else:  # 'continuous'
        i = numpy.where(m_rec[1:] != m_rec[:-1])[0]  # points where x axis (recall) changes
        ap = numpy.sum((m_rec[i + 1] - m_rec[i]) * m_pre[i + 1])  # area under curve

    return ap, m_pre, m_rec


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = numpy.zeros((nc, tp.shape[1])), numpy.zeros((nc, 1000)), numpy.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def strip_optimizer(f='best.pt'):
    x = torch.load(f, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, f)


def clip_gradient(parameters, clip_factor=35.0, norm_type=2.0):
    torch.nn.utils.clip_grad_norm_(parameters, clip_factor, norm_type)


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


def local_maximum(x, kernel=3):
    """
    Extract local maximum pixel with given kernel.
    Returns:
        A heatmap where local maximum pixels maintain its own value and other positions are 0.
    """
    x_max = max_pool2d(x, kernel, stride=1, padding=(kernel - 1) // 2)
    return x * (x_max == x).float()


def gaussian_target(heatmap, center, radius, k=1):
    """
    Generate 2D gaussian heatmap.
    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1

    x = torch.arange(-radius, radius + 1, dtype=heatmap.dtype, device=heatmap.device).view(1, -1)
    y = torch.arange(-radius, radius + 1, dtype=heatmap.dtype, device=heatmap.device).view(-1, 1)

    kernel = (-(x * x + y * y) / (2 * diameter / 6 * diameter / 6)).exp()

    kernel[kernel < torch.finfo(kernel.dtype).eps * kernel.max()] = 0

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = kernel[radius - top:radius + bottom, radius - left:radius + right]
    out_heatmap = heatmap
    torch.max(masked_heatmap,
              masked_gaussian * k,
              out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap


def gaussian_radius(det_size, min_overlap):
    """
    Generate 2D gaussian radius.
    Returns:
        radius (int): Radius of gaussian kernel.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def transpose_and_gather(x, indices):
    """
    Transpose and gather feature according to index.

    Args:
        x (Tensor): Target feature map.
        indices (Tensor): Target coord index.

    Returns:
        feature (Tensor): Transposed and gathered feature.
    """
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(x.size(0), -1, x.size(3))
    return x.gather(1, indices.unsqueeze(2).repeat(1, 1, x.size(2)))


class L1Loss(torch.nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, output, target,
                weight=1.0, average_factor=1.0):
        eps = torch.finfo(torch.float32).eps
        loss = torch.abs(output - target) * weight
        return self.loss_weight * loss.sum() / (average_factor + eps)


class GaussianFocalLoss(torch.nn.Module):
    def __init__(self, alpha=2.0, gamma=4.0, loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, output, target,
                weight=1.0, average_factor=1.0):
        epsilon = 1e-12

        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)

        neg_loss = -(1 - output + epsilon).log() * output.pow(self.alpha) * neg_weights
        pos_loss = -(output + epsilon).log() * (1 - output).pow(self.alpha) * pos_weights

        loss = (pos_loss + neg_loss) * weight
        return self.loss_weight * loss.sum() / (average_factor + torch.finfo(torch.float32).eps)


class ComputeLoss(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.loss_wh = L1Loss(0.1)
        self.loss_offset = L1Loss(1.0)
        self.loss_center = GaussianFocalLoss(1.0)

    def forward(self, outputs, targets):
        targets, average_factor = self.targets(targets, outputs[0].shape)
        # Since the channel of wh_target and offset_target is 2,
        # the avg_factor of loss_center is always 1/2 of loss_wh and loss_offset.
        loss_center = self.loss_center(outputs[0],
                                       targets[0],
                                       average_factor=average_factor)
        loss_wh = self.loss_wh(outputs[1], targets[1],
                               targets[3], average_factor * 2)
        loss_offset = self.loss_offset(outputs[2], targets[2],
                                       targets[3], average_factor * 2)
        return loss_center + loss_wh + loss_offset

    def targets(self, targets, shape):
        h, w = shape[2], shape[3]

        w_ratio = float(w / self.input_size)
        h_ratio = float(h / self.input_size)

        wh_size = (shape[0], 2, h, w)
        wh_true = torch.zeros(wh_size, dtype=torch.float32, device='cuda')

        offset_size = (shape[0], 2, h, w)
        offset_true = torch.zeros(offset_size, dtype=torch.float32, device='cuda')

        center_size = (shape[0], self.num_classes, h, w)
        center_true = torch.zeros(center_size, dtype=torch.float32, device='cuda')

        wh_offset_weight = torch.zeros([shape[0], 2, h, w], dtype=torch.float32).cuda()

        for i, target in enumerate(targets):
            boxes = target[:, 1:].cuda()
            label = target[:, :1].reshape(-1).long().cuda()

            center_x = (boxes[:, [0]] + boxes[:, [2]]) * w_ratio / 2
            center_y = (boxes[:, [1]] + boxes[:, [3]]) * h_ratio / 2

            centers = torch.cat((center_x, center_y), dim=1)
            for j, center in enumerate(centers):
                center_x, center_y = center
                c_x_int, c_y_int = center.int()

                scale_box_h = (boxes[j][3] - boxes[j][1]) * h_ratio
                scale_box_w = (boxes[j][2] - boxes[j][0]) * w_ratio

                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
                radius = max(0, int(radius))

                gaussian_target(center_true[i, label[j]], [c_x_int, c_y_int], radius)

                wh_true[i, 0, c_y_int, c_x_int] = scale_box_w
                wh_true[i, 1, c_y_int, c_x_int] = scale_box_h

                offset_true[i, 0, c_y_int, c_x_int] = center_x - c_x_int
                offset_true[i, 1, c_y_int, c_x_int] = center_y - c_y_int

                wh_offset_weight[i, :, c_y_int, c_x_int] = 1

        targets = (center_true, wh_true, offset_true, wh_offset_weight)
        return targets, max(1, center_true.eq(1).sum())


class Decoder:

    def __call__(self, center, wh, offset, input_size):
        outputs = []
        for i in range(center.shape[0]):
            x = center[i:i + 1, ...]
            y = wh[i:i + 1, ...]
            z = offset[i:i + 1, ...]

            batch_det_bboxes, batch_labels = self.decode(x, y, z,
                                                         input_size, k=100, kernel=3)

            det_bboxes = batch_det_bboxes.view([-1, 5])
            det_labels = batch_labels.view(-1)

            outputs.append(torch.cat((det_bboxes, det_labels[..., None]), -1))
        return outputs

    def decode(self, center, wh, offset,
               input_size, k=100, kernel=3):
        height, width = center.shape[2:]

        center = local_maximum(center, kernel)

        *batch, top_k_ys, top_k_xs = self.top_k(center, k=k)
        batch_scores, batch_index, batch_labels = batch

        wh = transpose_and_gather(wh, batch_index)
        offset = transpose_and_gather(offset, batch_index)

        top_k_xs = top_k_xs + offset[..., 0]
        top_k_ys = top_k_ys + offset[..., 1]

        tl_x = (top_k_xs - wh[..., 0] / 2) * (input_size / width)
        tl_y = (top_k_ys - wh[..., 1] / 2) * (input_size / height)
        br_x = (top_k_xs + wh[..., 0] / 2) * (input_size / width)
        br_y = (top_k_ys + wh[..., 1] / 2) * (input_size / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
        return batch_bboxes, batch_labels

    @staticmethod
    def top_k(scores, k=20):
        batch, _, height, width = scores.size()
        scores, indices = torch.topk(scores.view(batch, -1), k)
        classes = indices // (height * width)
        indices = indices % (height * width)
        top_y_s = indices // width
        top_x_s = (indices % width).int().float()
        return scores, indices, classes, top_y_s, top_x_s
