import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace
from utils import *
from network import ViT_model
import pickle

# from ours.Utils.utils import * # Georgios
# from ours.Networks.network import ViT_model # Georgios

### Setting arguments
args = SimpleNamespace(batch_size=1,
                       input_dim=224, 
                       pretrained_weights="path_to_ViTbase_imagenet",  ## from stored_weights we provided you with
                       ## pretrained on ImageNet taken from https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth'
                       val_set="path_to_ILSVRC2012_img_val, ## https://image-net.org/download-images.php download validation 2012 image all tasks
                       val_set_semg="path_to_gtsegs_ijcv.mat" ## downloaded from http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
                       labels_dict="path_to_val_labels_dict.npy", ## can be found in other folder
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                       )
dict = {}

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

transform_perturb = transforms.Compose([
    transforms.Resize((args.input_dim, args.input_dim)),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize((args.input_dim, args.input_dim)),
    transforms.ToTensor(),
    normalize,
])

transform_gt_mask = transforms.Compose([
    transforms.Resize((args.input_dim, args.input_dim)),
    transforms.ToTensor(),
])


# # # ## Constructing the validation loader
val_loader = ImageNetVal(args.val_set, args.labels_dict, args.device, transform_perturb) ## loading val split (50.000)
val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)

# Constructing the validation segm loader
val_loader_segm = ImageNetSegm(args.val_set_semg, args.device, transform, transform_gt_mask) ## loading val semgntation split (4.276)
val_loader_segm = DataLoader(val_loader_segm, batch_size=args.batch_size, shuffle=True)


## Initialize model
model = ViT_model(device=args.device)
model.load_pretrained(args.pretrained_weights)
model.eval()
model.zero_grad()

# Table 2.
pixAcc, mIoU, mAp = model.extract_metrics(val_loader_segm)
print(pixAcc, mIoU, mAp)


# Table 3.
# positive = True for positive perturbation
# positive = False for negative perturbation
# vis_class_top = True for positive predicted
# vis_class_top = False for negative target

AUC = model.extract_AUC(val_loader, normalize, positive=True, vis_class_top=True)
print(AUC)
