from torch.utils.data import Dataset
import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import math
import mat73
from skimage.transform import resize
import random
from iou import IoU
from metrices import *
from torch import nn


class ResizeMultipleChannels():

    def __init__(self, dim=(448,448), mode='bilinear'):
        self.dim = dim
        self.mode = mode


    def __call__(self, tensor):

        return torch.nn.Upsample(self.dim, mode=self.mode)(tensor.view(1, *tensor.shape))[0]


class RandomHorizontalFlip():

    def __init__(self,p=0.5):
        self.flip_p = p

    def __call__(self, img):
        if random.random() > (1-self.flip_p) :
            img = np.fliplr(img).copy()
        return img


class CHW_HWC():

    def __init__(self):
        pass

    def __call__(self,tensor):
        return tensor.permute(1,2,0)


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def pad_resize(img,desired_size):

    old_size = img.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = resize(img, new_size, preserve_range=True)

    new_im = np.zeros((desired_size,desired_size,3))
    new_im[int((desired_size - new_size[0]) // 2):(int((desired_size - new_size[0]) // 2)+img.shape[0]),
    int((desired_size - new_size[1]) // 2):(int((desired_size - new_size[1]) // 2)+img.shape[1]),:] = img

    img_window = [int((desired_size - new_size[0]) // 2),(int((desired_size - new_size[0]) // 2)+img.shape[0]),
                  int((desired_size - new_size[1]) // 2), (int((desired_size - new_size[1]) // 2)+img.shape[1])]

    return Image.fromarray(np.uint8(new_im)).convert('RGB'), img_window


def min_max_normalize(image):
    image_min = np.min(image)
    image_max = np.max(image)

    image = (image-image_min) / (image_max-image_min)# + 1e-6) ## safe division

    return image


class ImageNetVal(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_folder, labels_dict, device, transform):

        self.labels_dict = np.load(labels_dict, allow_pickle=True)
        self.transform = transform
        self.img_names = os.listdir(img_folder)
        self.img_names = np.asarray([img_folder+"/"+current_img for current_img in self.img_names])
        self.device = device

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):


        current_path = self.img_names[idx]
        img_id = current_path.split("/")[-1]

        img_orig = Image.open(current_path)
        n_channels = img_orig.layers

        if n_channels == 1:
            img_orig = img_orig.convert(mode='RGB')

        img = self.transform(img_orig)

        label = torch.IntTensor([self.labels_dict[img_id]])


        return img.to(self.device), label.to(self.device)


class ImageNetSegm:
    def __init__(self, path, device, transform_img, transform_seg_mask):
        # Load the data
        data = mat73.loadmat(path)
        data = data['value']

        # Store the device
        self.device = device

        # Extract arguments
        self.n_images = int(data['n'].item())
        self.images = data['img']
        self.image_ids = data['id']
        self.seg_masks = data['gt']

        # Specify transforms
        self.transform_img = transform_img
        self.transform_seg_mask = transform_seg_mask

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # The image with index idx
        img_orig = self.images[idx]
        if len(img_orig.shape) != 3:
            img_orig = img_orig.convert(mode='RGB')

        img_orig = Image.fromarray(img_orig)
        ## normalizing original image
        img_trans = self.transform_img(img_orig)

        ## resizing original image
        img_orig = self.transform_seg_mask(img_orig)

        ## resizing GT segmentation mask
        seg_mask_orig = self.seg_masks[idx]
        seg_mask_orig = Image.fromarray(seg_mask_orig[0])
        seg_mask_trans = self.transform_seg_mask(seg_mask_orig)


        return img_trans.to(self.device), seg_mask_trans.to(self.device), img_orig.to(self.device)



### Layer Initialization

def no_grad_trunc_normal_(tensor, mean=0, std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


######### metrics

def eval_batch(Res, labels):
    ## stolen from https://github.com/hila-chefer/Transformer-Explainability
    ## Thanks Hila Chefer

    # threshold between FG and BG is the mean
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1 - Res
    ######### ???????????????????????????????????
    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(0) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0.unsqueeze(0), Res_1.unsqueeze(0)), 0)
    output_AP = torch.cat((Res_0_AP.unsqueeze(0), Res_1_AP.unsqueeze(0)), 0)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label, batch_ap = 0, 0, 0, 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output.data.cpu(), labels[0, 0])
    ## labeled: all positive pixels in the groundtruth
    ## correct: all positive pixels in the groundtruth that were also predicted as positive (larger than the mean)

    inter, union = batch_intersection_union(output.data.cpu(), labels[0, 0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP.unsqueeze(0), labels[0]))
    batch_ap += ap

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, pred, target


class PascalVOC2012(Dataset):
    def __init__(self, img_names, labels_dict, voc12_img_folder, input_dim, device, transform=None):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.transform = transform
        self.input_dim = input_dim
        self.device = device


        with open(img_names) as file:
            self.img_paths = np.asarray([voc12_img_folder + l.rstrip("\n")+".jpg" for l in file])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]
        img_orig = Image.open(current_path)

        n_channels = img_orig.layers
        if n_channels == 1:
            img_orig = img_orig.convert(mode='RGB')

        img = self.transform(img_orig)

        ### resizing and padding the image to fix dimensions inputs
        orginal_shape = np.shape(img_orig)

        img_key = current_path.split("/")[-1][:-4]
        label = torch.from_numpy(self.labels_dict[img_key])

        return current_path, img.to(self.device), label.to(self.device), orginal_shape




class PascalVOC2012Affinity(Dataset):

    def __init__(self, img_names, voc12_img_folder, low_cams_folder, high_cams_folder,
                 input_dim, device, img_transform=None, label_transform=None, both_transform=None ):

        self.low_cams_folder = low_cams_folder
        self.high_cams_folder = high_cams_folder

        self.img_transform = img_transform
        self.label_transform = label_transform
        self.both_transform = both_transform

        self.input_dim = input_dim
        self.device = device

        self.avg_pool = nn.AvgPool2d((8, 8), stride=(8, 8))
        self.in_radius_labels = AffinityLabelExtraction(crop_size = input_dim//8, radius = 5)

        with open(img_names) as file:
            self.img_paths = np.asarray([voc12_img_folder + l.rstrip("\n")+".jpg" for l in file])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        img_key = current_path.split("/")[-1][:-4]

        img = Image.open(current_path)

        orginal_shape = np.shape(img)

        img = img.convert(mode='RGB')


        low_cam_path = self.low_cams_folder + img_key + ".npy"
        high_cam_path = self.high_cams_folder + img_key + ".npy"

        low_cam = np.load(low_cam_path, allow_pickle=True).item()
        high_cam = np.load(high_cam_path, allow_pickle=True).item()

        labels = np.asarray(list(low_cam.values()) + list(high_cam.values())).transpose(1, 2, 0)


        img = self.img_transform(img)
        labels = self.label_transform(labels)

        ## flipping both imgs and labels
        img_labels = torch.cat((img, labels), dim=0)
        if self.both_transform is not None:
            img_labels = self.both_transform(img_labels)

        img = img_labels[:3]
        labels = img_labels[3:]

        ## down-scaling labels
        labels = self.avg_pool(labels).permute(1,2,0)

        n_labels = labels.shape[-1]
        n_targets = n_labels // 2
        label_low =  labels[:, :, :n_targets]
        label_high =  labels[:, :, n_targets:]

        ## most confident label in each pixel
        label_low = torch.argmax(label_low, dim=-1).data.cpu().numpy()
        label_high = torch.argmax(label_high, dim=-1).data.cpu().numpy()
        label = label_low.copy()

        ## Note label_low: confident FG, unconfident BG
        ## Note label_high: confident BG, unconfident FG
        label[label_low == 0] = 255 ## unconfident BG = irrelevant region
        label[label_high == 0] = 0 ## confident BG = confident BG

        ### ExtractAffinityLabelInRadius
        labels = self.in_radius_labels(label)

        return current_path, img.to(self.device), labels, orginal_shape
#################################### CLASS ####################################

class AffinityLabelExtraction():

    def __init__(self, crop_size, radius=5, device="cuda"):
        self.radius = radius
        self.radius_floor = radius - 1 #might be useless
        self.device = device

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            # self.search_dist.append((0, y))
            for x in range(1-radius, radius):
                if self.euclidean(x, y) < math.pow(radius, 2):
                    self.search_dist.append((y, x))

        self.height = crop_size - radius + 1
        self.width = crop_size - 2 * radius + 2
        return

    def __call__(self, label):

        labels_from = label[:1 - self.radius, self.radius - 1:1 - self.radius]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []
        pix_max_val = 255

        for y, x in self.search_dist:
            labels_to = label[y:y + self.height, self.radius + x - 1:self.radius + x + self.width - 1]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, pix_max_val), np.less(labels_from, pix_max_val))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        ## stolen from https://github.com/jiwoon-ahn/psa
        ## Thanks Jiwoon Ahn
        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        ## stolen from https://github.com/jiwoon-ahn/psa
        ## Thanks Jiwoon Ahn
        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)
        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)
        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        positive_background = torch.from_numpy(bg_pos_affinity_label).to(self.device)
        positive_foreground = torch.from_numpy(fg_pos_affinity_label).to(self.device)
        negative_affinity = torch.from_numpy(neg_affinity_label).to(self.device)

        return positive_background, positive_foreground, negative_affinity


#################################### HELPERS ####################################

    def affinity_ce_losses(self,label_pred, affinities, count, index, tol=1e-5):
        if index == 2:
            affinities = 1 - affinities

        loss = - label_pred * torch.log(affinities + tol)
        loss = torch.sum(loss)
        loss /= count

        return loss

    def euclidean(self, x, y):
        return math.pow(x, 2) + math.pow(y, 2)

    def get_pairs_indices(self, radius, size):
        search_distances = []

        for x in range(1, radius):
            search_distances.append((0, x))

        for y in range(1, radius):
            # search_distances.append((0, y))
            for x in range(1-radius, radius):
                if euclidean(x, y) < math.pow(radius, 2):
                    search_distances.append((y, x))

        indices_whole = np.reshape(np.arange(0, size[-2]*size[-1], dtype=np.int64), (size[-2], size[-1]))

        indices_from = np.reshape(indices_whole[:1-radius, radius-1:1-radius], [-1])

        indices_result = []

        for y, x in search_distances:
            indices_to = indices_whole[y:y + 1 - radius + size[-2],
                         radius - 1 + x:(radius - 1) + x + size[-2] - 2*radius + 2]

            indices_to = np.reshape(indices_to, [-1])

            indices_result.append(indices_to)

        indices_result = np.concatenate(indices_result, axis = 0)

        return indices_from, indices_result



################### TODO implement ourselves

def _fast_hist(label_true, label_pred, n_class):

    # source https://github.com/Juliachang/SC-CAM
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)

    return hist


def scores(label_trues, label_preds, n_class):
    # https://github.com/Juliachang/SC-CAM

    hist = np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))


    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    ## stolen from https://github.com/jiwoon-ahn/psa
    ## Thanks Jiwoon Ahn


    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax


    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

class RandomCrop():
    ## stolen from https://github.com/jiwoon-ahn/psa
    ## Thanks Jiwoon Ahn

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]

        return container

