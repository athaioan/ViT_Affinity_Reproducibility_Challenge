import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace
from utils import *
from network import ViT_hybrid_model
import pickle
import os

### Setting arguments
args = SimpleNamespace(train_mode = True,
                       batch_size=1,
                       input_dim=448,
                       pretrained_weights="path_to_Hybrid_ViT_imagenet.pth", ## initialization -  from stored_weights we provided you with
                       # pretrained_weights="stored_weights/Hybrid_ViT_pascal.pth",  ## final - from stored_weights we provided you with
                       epochs=20,
                       lr=5e-3,
                       weight_decay=1e-4, 
                       VocClassList="path_to_PascalVocClasses.txt", ## can be found in other folder
                       voc12_img_folder="path_to_JPEGImages", ## download from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
                       gt_mask_fold = "path_to_VOC2012_SegmentationClass",

                       train_set="path_to_train_augm.txt",  ## can be found in other folder
                       val_set="path_to_val.txt",  ## can be found in other folder
                       labels_dict="path_to_cls_labels.npy", ## can be found in other folder
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       
                       
                       alpha_low = 4,
                       alpha_high = 32,
                       alpha_low_folder = "store_to_ViT_hybrid_crf_lows" ## set store folders
                       alpha_high_folder = "store_to_ViT_hybrid_crf_highs"
                       cam_folder = "store_to_ViT_hybrid_cams"
                       pred_folder = "store_to_ViT_hybrid_preds"      
                       
                       session_name = "set_session_name"
                       )


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_loader = PascalVOC2012(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim, args.device,
                              transform=transforms.Compose([
                              transforms.Resize((args.input_dim, args.input_dim)),
                              transforms.ToTensor(),
                              transforms.RandomHorizontalFlip(p=0.5),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                              normalize,
                              ]))

train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True)

val_loader = PascalVOC2012(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim, args.device,
                              transform=transforms.Compose([
                              transforms.Resize((args.input_dim, args.input_dim)),
                              transforms.ToTensor(),
                              normalize,
                              ]))
val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)

## Initialize model
model = ViT_hybrid_model(session_name=args.session_name, img_size=(448, 448), patch_size=16, n_heads=12, n_blocks=12,  embed_size=768, n_classes=20,
                         max_epochs=args.epochs, device=args.device)


model.load_pretrained(args.pretrained_weights)
model.eval()

if args.train_mode:

  # Prepare optimizer and scheduler
  optimizer = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=args.weight_decay)

  # Training ViT_Hybrid on Pascal Voc
  for index in range(model.max_epochs):

      for g in optimizer.param_groups:
          g['lr'] = args.lr * (1-index/model.max_epochs)

      print("Training epoch...")
      model.train_epoch(train_loader, optimizer)

      print("Validating epoch...")
      model.val_epoch(val_loader)

      model.visualize_graph()

      if model.val_history["loss"][-1] < model.min_val:
          print("Saving model...")
          model.min_val = model.val_history["loss"][-1]

          torch.save(model.state_dict(), model.session_name+"/stage_1.pth")
      

model.extract_LRP_for_affinity(train_loader, alpha_low=args.alpha_low, alpha_high=args.alpha_high,
                                 alpha_low_folder=os.path.join("train", args.alpha_low_folder), alpha_high_folder=os.path.join("train", args.alpha_high_folder),
                                 cam_folder=os.path.join("train", args.cam_folder), pred_folder=os.path.join("train", args.pred_folder))

model.extract_LRP_for_affinity(val_loader, alpha_low=args.alpha_low, alpha_high=args.alpha_high,
                                 alpha_low_folder=os.path.join("val", args.alpha_low_folder), alpha_high_folder=os.path.join("val", args.alpha_high_folder),
                                 cam_folder=os.path.join("val", args.cam_folder), pred_folder=os.path.join("val", args.pred_folder))


metrics = model.extract_mIoU(os.path.join("val", pred_folder), args.gt_mask_fold)

print(metrics)
