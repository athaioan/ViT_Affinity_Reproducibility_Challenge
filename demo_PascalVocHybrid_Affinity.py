import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace
from utils import *
from network import ViT_hybrid_model_Affinity
import pickle

### Setting arguments
args = SimpleNamespace(batch_size=8,
                       input_dim=448,
                       # pretrained_weights="path_to_Hybrid_ViT_pascal.pth",  ## from stored_weights we provided you with
                       pretrained_weights="path_to_Affinity_Hybrid_ViT_pascal.pth",  ## from stored_weights we provided you with
                       epochs=7,
                       lr=0.1,
                       weight_decay=1e-4,
                       VocClassList="path_to_PascalVocClasses.txt", ## can be found in other folder
                       voc12_img_folder="path_to_JPEGImages/", ## download from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
                       gt_mask_fold = "path_to_SegmentationClass/", ## download from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

                       train_set="path_to_train_augm.txt", ## can be found in other folder
                       val_set=r"path_to_val.txt",  ## can be found in other folder
                       low_cams_fold_val = "store_to_crf_lows/", ## set store folders
                       high_cams_fold_val = "store_to_crf_highs/", 
                       low_cams_fold_train="store_to_crf_lows/",
                       high_cams_fold_train="store_to_crf_highs/",
                       input_cams="store_to_cams/",
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       )


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_loader = PascalVOC2012Affinity(args.train_set,  args.voc12_img_folder, args.low_cams_fold_train, args.high_cams_fold_train,
                              args.input_dim, args.device,

                              img_transform=transforms.Compose([
                              transforms.Resize((args.input_dim, args.input_dim)),
                              transforms.ToTensor(),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                              normalize,
                              ]),

                             label_transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 ResizeMultipleChannels((args.input_dim, args.input_dim), mode='bilinear'),
                             ]),
                             both_transform=transforms.RandomHorizontalFlip(p=0.5))

train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True)


val_loader = PascalVOC2012Affinity(args.val_set,  args.voc12_img_folder, args.low_cams_fold_val, args.high_cams_fold_val,
                              args.input_dim, args.device,

                              img_transform=transforms.Compose([
                              transforms.Resize((args.input_dim, args.input_dim)),
                              transforms.ToTensor(),
                              normalize,
                              ]),
                             label_transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 ResizeMultipleChannels((args.input_dim, args.input_dim), mode='bilinear'),
                             ]))

val_loader = DataLoader(val_loader, batch_size=1, shuffle=False)


model = ViT_hybrid_model_Affinity(max_epochs=args.epochs, device=args.device)


model.load_pretrained(args.pretrained_weights)
model.session_name = "PascalVOC_classification_Hybrid_Affinity_1"

if not os.path.exists(model.session_name):
    os.makedirs(model.session_name)

#
# Prepare optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=args.weight_decay)

## Training AffinityNet on ViT_Hybrid explainability cues

# for index in range(model.max_epochs):
#
#     for g in optimizer.param_groups:
#         g['lr'] = args.lr * (1-index/model.max_epochs)
#
#     print("Training epoch...")
#     model.train_epoch(train_loader, optimizer)
#
#     print("Validating epoch...")
#     model.val_epoch(val_loader)
#
#     model.visualize_graph()
#
#     if model.val_history["loss"][-1] < model.min_val:
#         print("Saving model...")
#         model.min_val = model.val_history["loss"][-1]
#
#         torch.save(model.state_dict(), model.session_name+"/stage_1.pth")
#


model.affinity_refine_cams(val_loader,  args.input_dim, args.input_cams, pred_folder="aff_preds/",
                     alpha=4, beta=16, t_rw=8) # 4, 8 aff_preds1


cam_pred_fold = "C:/Users/johny/Desktop/Transformer-Explainability-main/ours/aff_preds/" ## path to aff_preds


metrics = model.extract_mIoU(cam_pred_fold, gt_mask_fold)

print(metrics)


