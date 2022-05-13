import torch
from torch import nn
import matplotlib.pyplot as plt
from overwritten_layers import *
from utils import *
from einops import rearrange
import imageio
import os
from collections import *
from functools import partial
import collections
from itertools import repeat


class ViT_model(nn.Module):
    def __init__(self, n_classes=1000, img_size=(224, 224), patch_size=16, in_ch=3, embed_size=768,
                 n_heads=12, QKV_bias=True, att_dropout=0., out_dropout=0., n_blocks=12, mlp_hidden_ratio=4.,
                 device="cuda", max_epochs=10):
        super(ViT_model, self).__init__()

        self.train_history = {"loss": []}
        self.val_history = {"loss": []}
        self.min_val = np.inf

        self.current_epoch = 0
        self.max_epochs = max_epochs

        self.device = device
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.QKV_bias = QKV_bias

        self.add = Add()
        self.patch_embed = Img_to_patch(img_size, patch_size, in_ch, embed_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))

        set_seeds(0)
        no_grad_trunc_normal_(self.pos_embed, std=.02)
        no_grad_trunc_normal_(self.cls_token, std=.02)

        self.input_grad = None

        self.blocks = nn.ModuleList([Block(embed_size=self.embed_size, n_heads=self.n_heads,
                           QKV_bias=self.QKV_bias, att_dropout=att_dropout, out_dropout=out_dropout, mlp_hidden_ratio=4)
                                       for _ in range(self.n_blocks)])

        self.norm = LayerNorm(embed_size)
        self.head = Linear(self.embed_size, self.n_classes)

        self.pool = ClsSelect()

        self.to(self.device)


    def forward(self, x):

        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1)# from Phil Wang
        x = torch.cat((cls_token, x), dim=1)
        x = self.add([x, self.pos_embed])# x+= self.positional_embed

        if x.requires_grad:
            x.register_hook(self.store_input_grad) ## When computing the grad wrt to the input x, store that grad to the model.input_grad

        for current_block in self.blocks:
            x = current_block(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, index=torch.tensor(0, device=x.device)) ## retrieve the cls
        x = x.squeeze(1)
        x = self.head(x)

        return x

    def store_input_grad(self, grad):
        self.input_grad = grad

    def compute_att_rollout(self, all_relevances):
        b = 1 # only batch size of one is supported
        n = all_relevances[-1].shape[0]
        I = torch.eye(n).to(self.device)
        ## eq 13
        A = [all_relevances[index]+I for index in range(len(all_relevances))]

        att_rollout = A[0]
        for index in range(1,len(all_relevances)):
            att_rollout = A[index].matmul(att_rollout)
        return att_rollout


    def relevance_propagation(self, one_hot_label):

        ## from top to bottom
        relevance = self.head.relevance_propagation(one_hot_label)
        relevance = self.pool.relevance_propagation(relevance)
        relevance = self.norm.relevance_propagation(relevance)

        for current_block in reversed(self.blocks):
            relevance = current_block.relevance_propagation(relevance)

        all_relevances = []
        ## transformer_attribution
        for current_block in self.blocks:
            current_grad = current_block.attn.get_att_grad()
            current_relevance = current_block.attn.get_att_relevance()
            current_grad = current_grad.squeeze(0)
            current_relevance = current_relevance.squeeze(0)
            current_relevance *= current_grad

            ## considering only (+)
            current_relevance = current_relevance.clamp(min=0)

            ## averaging across the head dimension in accordance to Eq. 13
            current_relevance = current_relevance.mean(dim=0)

            all_relevances.append(current_relevance)

        att_rollout = self.compute_att_rollout(all_relevances)

        ## cls token
        att_rollout = att_rollout[0,1:]

        return att_rollout


    def extract_LRP(self, input, class_indices = None, root=1):

        pred = self(input)

        if class_indices is None:
            class_indices = torch.argmax(pred, dim=1).data.cpu().numpy().tolist()


        one_hot = np.zeros((1, pred.shape[-1]), dtype=np.float32)
        one_hot[0, class_indices] = 1

        one_hot_label = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(input.device) * pred)

        self.zero_grad()
        one_hot.backward(retain_graph=True) ## Register_hooks are excecuted in here

        att_rollout = self.relevance_propagation(torch.tensor(one_hot_label).to(input.device))


        ## reshaping att_rollout
        cue_size = int(self.patch_embed.n_patches ** (0.5))
        explainability_cue = att_rollout.reshape(1, 1,cue_size, cue_size)

        ## scaling to input's dimensions
        input_size = self.img_size[0]
        explainability_cue = torch.nn.functional.interpolate(
            explainability_cue, scale_factor=input_size//cue_size, mode='bilinear')[0,0]


        explainability_cue = explainability_cue.data.cpu().numpy()

        explainability_cue = explainability_cue**(1/root)

        explainability_cue = min_max_normalize(explainability_cue)

        explainability_cue = torch.from_numpy(explainability_cue)

        return explainability_cue.to(self.device), pred

    def train_epoch(self, dataloader, optimizer):


        train_loss = 0
        self.train()
        self.current_epoch += 1

        for index, data in enumerate(dataloader):


            img = data[1]
            label = data[2]

            x = self(img)
            # explainability_cue, preds = self.extract_LRP(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss.item()

            ### Printing epoch results
            print('Train Epoch: {}/{}\n'
                  'Step: {}/{}\n'
                  'Batch ~ Loss: {:.4f}\n'
                  .format(self.current_epoch, self.max_epochs,
                          index + 1, len(dataloader),
                          train_loss / (index + 1)))

        self.train_history["loss"].append(train_loss / len(dataloader))
        return

    def val_epoch(self, dataloader):

        val_loss = 0
        self.eval()
        with torch.no_grad():

            for index, data in enumerate(dataloader):

                img = data[1]
                label = data[2]

                x = self(img)

                loss = F.multilabel_soft_margin_loss(x, label)

                ### adding batch loss into the overall loss
                val_loss += loss.item()

                ### Printing epoch results
                print('Val Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      .format(self.current_epoch, self.max_epochs,
                              index + 1, len(dataloader),
                              val_loss/(index+1)))

            self.val_history["loss"].append(val_loss / len(dataloader))
        return

    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(os.path.join(session_name,"/loss.png"))
        plt.close()
        return

    def load_pretrained(self, weights_path):

        ## loading weights
        weights_dict = torch.load(weights_path)

        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}

        no_pretrained_dict = {k: v for k, v in model_dict.items() if
                           not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


    def extract_metrics(self, dataloader, vis_class_top=True):
        ## stolen from https://github.com/hila-chefer/Transformer-Explainability
        ## Thanks Hila Chefer

        predictions, targets = [], []
        total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
        total_ap = []

        for index, data in enumerate(dataloader):

            print("Evaluating Segmentation Val set ",index," out of ", len(dataloader))

            img = data[0]
            label = data[1]

            vis_class = None if vis_class_top else label[0,0].data.cpu().numpy().tolist()
            explainability_cue, preds = self.extract_LRP(img, class_indices=vis_class)

            # # create heatmap from mask on image
            # import cv2
            # def show_cam_on_image(img, mask):
            #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            #     heatmap = np.float32(heatmap) / 255
            #     cam = heatmap + np.float32(img)
            #     cam = cam / np.max(cam)
            #     return cam
            #
            # c = show_cam_on_image(data[2][0].data.cpu().numpy().transpose(1, 2, 0),
            #                       explainability_cue.data.cpu().numpy())
            # # c = show_cam_on_image(data[2][0].data.cpu().numpy().transpose(1, 2, 0)/255,
            # # np.ones_like(explainability_cue.data.cpu().numpy()))
            #
            # vis = np.uint8(255 * c)
            # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            #
            # plt.imshow(vis)
            # plt.tight_layout(pad=0)
            # plt.axis('off')
            # plt.savefig('heat_map.png')
            #
            # plt.imshow(data[2][0].data.cpu().numpy().transpose(1, 2, 0))
            # plt.tight_layout(pad=0)
            # plt.axis('off')
            # plt.savefig('gt.png')
            #


            correct, labeled, inter, union, ap, pred, target = eval_batch(explainability_cue, label)

            predictions.append(pred)
            targets.append(target)

            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            total_ap += [ap]
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            mAp = np.mean(total_ap)

        return pixAcc, mIoU, mAp

    def extract_AUC(self, dataloader, transform, positive=False, vis_class_top=True):

        ## positive : True = when doing the positive perturbation and
        ##            False = when doing the negative
        ## vis_class_target: True = when extracting the explainability cue wrt the target class
        ##                   False  = when extracting the explainability cue wrt the predicted class

        pred_accuracy = np.zeros(10)

        for index, data in enumerate(dataloader):

            print("Evaluating AUC in Val set ",index," out of ", len(dataloader))


            img = data[0]
            img_ = transform(img)
            label = data[1]

            vis_class = None if vis_class_top else label[0,0].data.cpu().numpy().tolist()
            explainability_cue, preds = self.extract_LRP(img_, class_indices=vis_class)

            pred_c = torch.argmax(preds).data.cpu().numpy()
            target_c = label[0,0].data.cpu().numpy()
            ####
            pred_accuracy[0] += pred_c == target_c

            if not positive:
                ## negative
                explainability_cue = - explainability_cue

            explainability_cue = explainability_cue.flatten()
            N_pixels = len(explainability_cue)

            for current_step in range(1, 10):

                current_img = img.clone() ## copying img
                _, indices_perturb = torch.topk(explainability_cue, int(N_pixels * current_step/10))
                indices_perturb = indices_perturb.repeat((3, 1)).unsqueeze(0)
                current_img = current_img.flatten(start_dim=-2, end_dim=-1)
                current_img = current_img.scatter_(-1, indices_perturb, 0)
                current_img = current_img.reshape(img.size())

                current_img = transform(current_img)

                current_preds = self(current_img)
                current_pred_c = torch.argmax(current_preds).data.cpu().numpy()
                pred_accuracy[current_step] += current_pred_c == target_c

        pred_accuracy /= len(dataloader)
        AUC = np.trapz(pred_accuracy, dx=0.1)

        return AUC

    def extract_mIoU(self, cam_folder, gt_folder):

        cam_paths = os.listdir(cam_folder)

        label_trues = []
        label_preds = []
        for index, current_cam_key in enumerate(cam_paths):

            print("Step",index/len(cam_paths))



            current_cam_path = cam_folder + "/" + current_cam_key
            current_gt_path = gt_folder + "/" + current_cam_key

            current_cam_pred = Image.open(current_cam_path)
            current_cam_pred = np.array(current_cam_pred)

            current_gt = Image.open(current_gt_path)
            current_gt = np.array(current_gt)

            label_preds.append(current_cam_pred)
            label_trues.append(current_gt)

        metrics = scores(label_trues, label_preds, self.n_classes+1)

        return metrics



    def extract_LRP_for_affinity(self, dataloader, alpha_low=4, alpha_high=32,
                                 alpha_low_folder = "crf_lows2/", alpha_high_folder = "crf_highs2/",
                                 cam_folder = "cams2/", pred_folder = "preds2/"):

        if not os.path.exists(alpha_low_folder):
            os.makedirs(alpha_low_folder)

        if not os.path.exists(alpha_high_folder):
            os.makedirs(alpha_high_folder)

        if not os.path.exists(cam_folder):
            os.makedirs(cam_folder)

        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)


        for index, data in enumerate(dataloader):

            ## Extracting explainability cue and prediction
            print("Extracting Explainability cue and Prediction",index," out of ", len(dataloader))

            img_key = data[0][0].split("/")[-1][:-4]
            img = data[1]
            label = data[2]

            img_orig = plt.imread(data[0][0])
            img_orig_size = [i[0].data.cpu().numpy() for i in data[-1]]

            explainability_pred = np.zeros((self.n_classes, img_orig_size[0],
                                          img_orig_size[1]))

            explainability_bg = np.ones((img_orig_size[0],
                                          img_orig_size[1]))*0.2

            explainability_LRPs = {}
            vis_class = np.nonzero(label[0].data.cpu().numpy())[0]
            for current_vis_class in vis_class:

                for flip_flag in [0,1]:

                    if flip_flag:
                        current_explainability_cue_flipped, preds = self.extract_LRP(torch.flip(img, (0,3)) , class_indices=current_vis_class, root=3)

                        current_explainability_cue_flipped = torch.nn.Upsample((img_orig_size[0], img_orig_size[1]),
                                                                       mode='bilinear') \
                            (current_explainability_cue_flipped.view(1, 1, *current_explainability_cue_flipped.shape))

                        current_explainability_cue_flipped = torch.flip(current_explainability_cue_flipped, (0,3))

                        current_explainability_cue+=current_explainability_cue_flipped
                        current_explainability_cue /=2
                    else:
                        current_explainability_cue, preds = self.extract_LRP(img,
                                                                             class_indices=current_vis_class, root=3)

                        current_explainability_cue = torch.nn.Upsample((img_orig_size[0], img_orig_size[1]),
                                                                        mode='bilinear') \
                            (current_explainability_cue.view(1, 1, *current_explainability_cue.shape))

                explainability_LRPs[current_vis_class] = current_explainability_cue.data.cpu().numpy()[0,0]
                explainability_pred[current_vis_class] = current_explainability_cue.data.cpu().numpy()[0,0]

            explainability_pred = np.concatenate((explainability_bg[None,...], explainability_pred))
            explainability_pred = np.argmax(explainability_pred,axis=0)

            ## save cam
            np.save(cam_folder + img_key + ".npy", explainability_LRPs)

            ## pred
            imageio.imwrite(pred_folder + img_key + ".png", explainability_pred.astype(np.uint8))

            ### confident foreground
            LRP_v = np.array(tuple(explainability_LRPs.values()))
            bg_v = (1 - np.max(LRP_v,axis=0))**alpha_low

            v = np.concatenate((bg_v[None,...],LRP_v),axis=0)
            crf_low = crf_inference(img_orig, v, labels=LRP_v.shape[0]+1)

            crf_low_dict = {}
            crf_low_dict[0] = crf_low[0]
            for index, current_class in enumerate(vis_class):
                crf_low_dict[current_class+1] = crf_low[index+1]

            np.save(alpha_low_folder+img_key+".npy", crf_low_dict)

            ### confident background
            LRP_v = np.array(tuple(explainability_LRPs.values()))
            bg_v = (1 - np.max(LRP_v, axis=0)) ** alpha_high

            v = np.concatenate((bg_v[None, ...], LRP_v), axis=0)
            crf_high = crf_inference(img_orig, v, labels=LRP_v.shape[0] + 1)

            crf_high_dict = {}
            crf_high_dict[0] = crf_high[0]
            for index, current_class in enumerate(vis_class):
                crf_high_dict[current_class + 1] = crf_high[index + 1]

            np.save(alpha_high_folder + img_key + ".npy", crf_high_dict)



class Img_to_patch(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, input_ch=3, embed_size=768):
        super(Img_to_patch, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_ch = input_ch
        self.embed_size = embed_size
        ## TODO architecture
        self.proj = Conv2d(self.input_ch, self.embed_size, kernel_size=(self.patch_size, self.patch_size),
                             stride=(self.patch_size, self.patch_size))

        self.n_patches = (img_size[1] // self.patch_size) * (img_size[0] // self.patch_size)

    def forward(self, x):

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention_layer(nn.Module):

    def __init__(self, embed_size=768, n_heads=12, QKV_bias=False, att_dropout=0., out_dropout=0.):
        super().__init__()

        self.n_heads = n_heads
        self.QKV_bias = QKV_bias

        head_dim = embed_size // n_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = Linear(embed_size, embed_size * 3, bias=self.QKV_bias)
        self.proj = Linear(embed_size, embed_size)

        # A = Q*K^T
        self.matmul1 = Matmul(transpose=True)
        # att = A*V
        self.matmul2 = Matmul(transpose=False)
        self.att_softmax = Softmax(dim=-1)


        self.att_dropout = Dropout(att_dropout)
        self.out_dropout = Dropout(out_dropout)

        self.v = None
        self.att = None
        self.att_grad = None

        self.v_relevance = None
        self.att_relevance = None

    # TODO eliminated those not being used

    def store_v(self, v):
        self.v = v

    def store_att(self, att):
        self.att = att

    def store_att_grad(self, grad):
        self.att_grad = grad

    def store_v_relevance(self, relevance):
        self.v_relevance = relevance

    def store_att_relevance(self, relevance):
        self.att_relevance = relevance

    def get_v(self):
        return self.v

    def get_att(self):
        return self.att

    def get_att_grad(self):
        return self.att_grad

    def get_v_relevance(self):
        return self.v_relevance

    def get_att_relevance(self):
        return self.att_relevance


    def forward(self, x):
        batch, n, embed_size = x.shape
        qkv = self.qkv(x)

        ## ours
        qkv = torch.reshape(qkv, (batch, n, 3, self.n_heads, embed_size//self.n_heads))
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        self.store_v(v)

        # A = Q * K.T
        scaled_products = self.matmul1([q, k]) * self.scale

        att = self.att_softmax(scaled_products)
        att = self.att_dropout(att)

        self.store_att(att)

        if att.requires_grad:
            att.register_hook(self.store_att_grad)

        # att = A*V
        x = self.matmul2([att, v])
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (batch, n, embed_size))


        x = self.proj(x)
        x = self.out_dropout(x)

        return x

    def relevance_propagation(self, relevance):
        batch, n, embed_size = relevance.shape

        relevance = self.out_dropout.relevance_propagation(relevance)
        relevance = self.proj.relevance_propagation(relevance)

        relevance = torch.reshape(relevance,
                            (batch, n, self.n_heads, embed_size//self.n_heads))
        relevance = relevance.permute(0, 2, 1, 3)


        relevance, relevance_v = self.matmul2.relevance_propagation(relevance)
        ## TODO why? /2
        relevance /=2
        relevance_v /=2

        self.store_v_relevance(relevance_v)
        self.store_att_relevance(relevance)

        relevance = self.att_dropout.relevance_propagation(relevance)
        relevance = self.att_softmax.relevance_propagation(relevance)

        relevance_q, relevance_k = self.matmul1.relevance_propagation(relevance)

        ## TODO why? /2
        relevance_q /=2
        relevance_k /=2

        relevance_qkv = torch.stack([relevance_q,
                                   relevance_k,
                                   relevance_v])

        relevance_qkv = relevance_qkv.permute(1, 3, 0, 2, 4)
        relevance_qkv = torch.reshape(relevance_qkv, (batch, n, 3*embed_size))

        relevance_qkv = self.qkv.relevance_propagation(relevance_qkv)

        return relevance_qkv

class Mlp(nn.Module):

    def __init__(self, in_dim, hidden_dim=None, dropout=0.):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_dim

        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, in_dim)
        self.dropout = Dropout(dropout)
        self.gelu = GELU()



    def forward(self, x):
        ## FC1
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        ## FC2
        x = self.fc2(x)
        x = self.dropout(x)

        return x

    def relevance_propagation(self, relevance):
        ## FC2
        relevance = self.dropout.relevance_propagation(relevance)
        relevance = self.fc2.relevance_propagation(relevance)

        ## FC1
        relevance = self.dropout.relevance_propagation(relevance)
        relevance = self.gelu.relevance_propagation(relevance)
        relevance = self.fc1.relevance_propagation(relevance)

        return relevance

class Block(nn.Module):

    def __init__(self, embed_size=768, n_heads=12, QKV_bias=True, att_dropout=0., out_dropout=0.,
                 mlp_hidden_ratio=4):
        super().__init__()

        self.embed_size = embed_size
        self.n_heads = n_heads
        self.QKV_bias = QKV_bias
        self.mlp_hidden_ratio = mlp_hidden_ratio


        ## MLP layer
        self.mlp = Mlp(embed_size, hidden_dim=int(self.mlp_hidden_ratio*self.embed_size), dropout=out_dropout)

        ## Attention layer
        self.attn = Attention_layer(embed_size=self.embed_size, n_heads=self.n_heads, QKV_bias=self.QKV_bias,
                                   att_dropout=att_dropout, out_dropout=out_dropout)

        ## Normalization layers
        self.norm1 = LayerNorm(self.embed_size, eps=1e-6)
        self.norm2 = LayerNorm(self.embed_size, eps=1e-6)

        self.add1 = Add()
        self.add2 = Add()

        self.clone1 = Clone()
        self.clone2 = Clone()

    ###### GM NEW ###### todo --> remove comment after explaining
    def relevance_propagation(self, relevance):
        (relevance, relevance_dupl) = self.add2.relevance_propagation(relevance)
        relevance_dupl = self.mlp.relevance_propagation(relevance_dupl)
        relevance_dupl = self.norm2.relevance_propagation(relevance_dupl)
        relevance = self.clone2.relevance_propagation((relevance, relevance_dupl))

        (relevance, relevance_dupl) = self.add1.relevance_propagation(relevance)
        relevance_dupl = self.attn.relevance_propagation(relevance_dupl)
        relevance_dupl = self.norm1.relevance_propagation(relevance_dupl)
        relevance = self.clone1.relevance_propagation((relevance, relevance_dupl))

        return relevance



    def forward(self,x):

        x1, x2 = self.clone1(x, 2)
        x2 = self.norm1(x2)
        x2 = self.attn(x2)
        x = self.add1([x1, x2])

        x1, x2 = self.clone2(x, 2)
        x2 = self.norm2(x2)
        x2 = self.mlp(x2)
        x = self.add2([x1, x2])

        return x



class ViT_hybrid_model(ViT_model):

    def __init__(self, session_name, n_classes=1000, img_size=(224, 224), patch_size=16, in_ch=3, embed_size=768,
                 n_heads=12, QKV_bias=True, att_dropout=0., out_dropout=0., n_blocks=12, mlp_hidden_ratio=4.,
                 device="cuda", max_epochs=10):

        super(ViT_hybrid_model, self).__init__(n_classes=n_classes, img_size=img_size, patch_size=patch_size, in_ch=in_ch, embed_size=embed_size,
                 n_heads=n_heads, QKV_bias=QKV_bias, att_dropout=att_dropout, out_dropout=out_dropout, n_blocks=n_blocks, mlp_hidden_ratio=mlp_hidden_ratio,
                 device=device, max_epochs=max_epochs)

        self.session_name = session_name
        
        if not os.path.exists(model.session_name):
            os.makedirs(model.session_name)
        
        # self.resnet_backbone = ResNetV2(block_units=(3, 4, 9),
        #                                  width_factor=1)
        self.resnet_backbone = ResNetV2(
            layers=(3, 4, 9), num_classes=0, global_pool='', in_chans=3,
            preact=False, stem_type='same', conv_layer=StdConv2dSame)

        self.in_ch = 1024

        self.patch_size = 1

        self.patch_embed = Img_to_patch((1,1), 1, self.in_ch , embed_size)

        self.patch_embed.n_patches = (img_size[0] // 16) * (img_size[1] // 16)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))

        self.to(self.device)


    def forward(self, x):

        batch_size = x.shape[0]

        feat = self.resnet_backbone(x)

        x = self.patch_embed(feat)

        cls_token = self.cls_token.expand(batch_size, -1, -1)# from Phil Wang
        x = torch.cat((cls_token, x), dim=1)
        x = self.add([x, self.pos_embed])# x+= self.positional_embed

        if x.requires_grad:
            x.register_hook(self.store_input_grad) ## When computing the grad wrt to the input x, store that grad to the model.input_grad

        for current_block in self.blocks:
            x = current_block(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, index=torch.tensor(0, device=x.device)) ## retrieve the cls
        x = x.squeeze(1)
        x = self.head(x)

        return x



class ViT_hybrid_model_Affinity(nn.Module):

    def __init__(self, session_name, max_epochs=10, device="cuda", n_classes=20):

        super(ViT_hybrid_model_Affinity, self).__init__()
        self.train_history ={}
        self.val_history ={}

        self.train_history["loss"] = []
        self.train_history["fg_loss"] = []
        self.train_history["bg_loss"] = []
        self.train_history["neg_loss"] = []

        self.val_history["loss"] = []
        self.val_history["fg_loss"] = []
        self.val_history["bg_loss"] = []
        self.val_history["neg_loss"] = []

        
        self.min_val = np.inf

        self.n_classes = n_classes
        
        self.session_name = session_name
        if not os.path.exists(model.session_name):
            os.makedirs(model.session_name)
        
        
        self.device = device
        self.max_epochs = max_epochs
        self.current_epoch = 0

        self.resnet_backbone = ResNetV2(
            layers=(3, 4, 9), num_classes=0, global_pool='', in_chans=3,
            preact=False, stem_type='same', conv_layer=StdConv2dSame)


        self.feat_conv = torch.nn.Conv2d(1024, 512, 1, bias=False)
        self.gn = nn.modules.normalization.GroupNorm(32, 512)
        self.affinity_conv = torch.nn.Conv2d(512, 512, 1, bias=False)

        self.predefined_grid = 56


        torch.nn.init.kaiming_normal_(self.feat_conv.weight)
        torch.nn.init.xavier_uniform_(self.affinity_conv.weight, gain=4)

        self.indices_from, self.indices_to = self.get_pairs_indices(5, (self.predefined_grid,self.predefined_grid))
        self.indices_from = torch.from_numpy(self.indices_from).to(self.device)
        self.indices_to = torch.from_numpy(self.indices_to).to(self.device)

        self.avg_pool = nn.AvgPool2d((8, 8), stride=(8, 8))


        self.to(self.device)


    def forward(self, x, val_mode=False):

        feat = self.resnet_backbone(x)

        x = torch.nn.Upsample((self.predefined_grid, self.predefined_grid), mode='bilinear')(feat)

        x = self.feat_conv(x)
        x = self.gn(x)
        x = F.elu(x)

        x = self.affinity_conv(x)
        x = F.elu(x)

        ### predefined feature size
        x = x.view(x.size(0), x.size(1), -1)

        feature_from = torch.index_select(x, dim=2,
                                          index=self.indices_from)

        feature_to = torch.index_select(x, dim=2,
                                          index=self.indices_to)

        feature_from = torch.unsqueeze(feature_from, dim=2)

        feature_to = feature_to.view(feature_to.size(0),
                                     feature_to.size(1), -1, feature_from.size(3))

        aff = torch.exp(-torch.mean(torch.abs(feature_from-feature_to), dim=1))

        if val_mode:
            ## returning affinity as flatten array
            aff = aff.view(-1) ## flattening the predicted affinity array

            indices_from_vs_indices_to = torch.unsqueeze(self.indices_from, dim=0) \
            .expand(feature_to.size(2), -1).contiguous().view(-1)

            indices_t = torch.stack([indices_from_vs_indices_to, self.indices_to]).to(self.device)

            indices_tf = torch.stack([self.indices_to, indices_from_vs_indices_to]).to(self.device)

            area = x.size(-1)
            indices_ones = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()]).to(self.device)

            aff_mat = torch.sparse.FloatTensor(torch.cat([indices_t, indices_ones, indices_tf], dim=1),
                                      torch.cat([aff, torch.ones([area]).to(self.device), aff])).to_dense()\
                .to(self.device)

            return aff_mat

        else:

            return aff

    def extract_mIoU(self, cam_folder, gt_folder):

        cam_paths = os.listdir(cam_folder)

        label_trues = []
        label_preds = []
        for index, current_cam_key in enumerate(cam_paths):

            print("Computing mIoU ",index," out of ", len(cam_paths))


            current_cam_path = cam_folder + "/" + current_cam_key
            current_gt_path = gt_folder + "/" + current_cam_key

            current_cam_pred = Image.open(current_cam_path)
            current_cam_pred = np.array(current_cam_pred)

            current_gt = Image.open(current_gt_path)
            current_gt = np.array(current_gt)

            label_preds.append(current_cam_pred)
            label_trues.append(current_gt)

        metrics = scores(label_trues, label_preds, self.n_classes+1)

        return metrics


    def affinity_refine_cams(self, dataloader, input_dim, cam_input_folder, pred_folder="aff_preds/",
                             alpha=4, beta=8, t_rw=8):

        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)

        for index, data in enumerate(dataloader):

            print("Refining cams ",index," out of ", len(dataloader))


            img_key = data[0][0].split("/")[-1][:-4]
            img = data[1]
            # labels = data[2]
            orig_size = data[3]

            cam_path = cam_input_folder+img_key+".npy"
            cam_dict = np.load(cam_path, allow_pickle=True).item()

            cam_array = np.zeros((21, orig_size[0], orig_size[1]))
            for key, current_cam in cam_dict.items():
                ## plus one to account for the BG_class = 0
                cam_array[key+1] =  current_cam

            BG_cam = (1-np.max(cam_array, axis=0))**alpha
            cam_array[0] = BG_cam

            cam_array = torch.from_numpy(cam_array).to(self.device)
            cam_array = torch.nn.Upsample(input_dim, mode="bilinear")(cam_array.view(1, *cam_array.shape))[0]

            dense_aff = self(img, val_mode=True).double()

            dense_aff = dense_aff ** beta ## filtering-out insignificant relationships

            ## obtaining the transition matrix through
            trans_mat = dense_aff / torch.sum(dense_aff,dim=0)

            ## applying the randomwalk multiple times
            for i in range(t_rw):
                trans_mat = torch.matmul(trans_mat, trans_mat)


            cam_array = self.avg_pool(cam_array) ## transforming the cam into the featature map dims

            cam_array_flatten = cam_array.view(self.n_classes+1, -1)

            cam_rw_refined = torch.matmul(cam_array_flatten, trans_mat)

            cam_rw_refined = cam_rw_refined.view(1, self.n_classes+1, self.predefined_grid, self.predefined_grid)

            cam_rw_refined = torch.nn.Upsample((orig_size[0],orig_size[1]), mode='bilinear')(cam_rw_refined)[0]

            pred = np.argmax(cam_rw_refined.data.cpu().numpy(),axis=0)


            imageio.imwrite(pred_folder + img_key + ".png", pred.astype(np.uint8))


    def euclidean(self, x, y):
        return math.pow(x, 2) + math.pow(y, 2)

    def get_pairs_indices(self, radius=5, size=(56, 56)):

        self.in_radius = radius-1

        search_distances = []

        for x in range(1, radius):
            search_distances.append((0, x))

        for y in range(1, radius):
            # search_distances.append((0, y))
            for x in range(1 - radius, radius):
                if self.euclidean(x, y) < math.pow(radius, 2):
                    search_distances.append((y, x))

        indices_whole = np.reshape(np.arange(0, size[-2] * size[-1], dtype=np.int64), (size[-2], size[-1]))

        indices_from = np.reshape(indices_whole[:1 - radius, radius - 1:1 - radius], [-1])

        indices_result = []

        for y, x in search_distances:
            indices_to = indices_whole[y:y + 1 - radius + size[-2],
                         radius - 1 + x:(radius - 1) + x + size[-2] - 2 * radius + 2]

            indices_to = np.reshape(indices_to, [-1])

            indices_result.append(indices_to)

        indices_result = np.concatenate(indices_result, axis=0)

        return indices_from, indices_result

    def train_epoch(self, dataloader, optimizer):

        train_loss = 0
        train_fg_loss = 0
        train_bg_loss = 0
        train_neg_loss = 0
        self.train()
        self.current_epoch += 1


        for index, data in enumerate(dataloader):


            img = data[1]
            labels = data[2]

            aff_pred = self(img)


            BG_labels = labels[0]
            FG_labels = labels[1]
            NEG_labels = labels[2]

            n_bg_affinities = torch.sum(BG_labels) + 1e-6
            n_fg_affinities = torch.sum(FG_labels) + 1e-6
            n_neg_affinities = torch.sum(NEG_labels) + 1e-6

            ## loss according to Eq. 7,8,9,10
            ## BG loss
            fg_loss = torch.sum(-FG_labels * torch.log(aff_pred+1e-6)) / n_fg_affinities
            bg_loss = torch.sum(-BG_labels * torch.log(aff_pred+1e-6)) / n_bg_affinities
            neg_loss = torch.sum(-NEG_labels * torch.log(1 - aff_pred + 1e-6)) / n_neg_affinities


            # explainability_cue, preds = self.extract_LRP(img)
            loss = fg_loss/4 + bg_loss/4 + neg_loss/2
            ##

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss.item()
            train_fg_loss += fg_loss.item()
            train_bg_loss += bg_loss.item()
            train_neg_loss += neg_loss.item()

            ### Printing epoch results
            print('Train Epoch: {}/{}\n'
                  'Step: {}/{}\n'
                  'Batch ~ Overall Loss: {:.4f}\n'
                  'Batch ~ FG Loss: {:.4f}\n'
                  'Batch ~ BG Loss: {:.4f}\n'
                  'Batch ~ NEG Loss: {:.4f}\n'
                  'Batch ~ FG Count: {:.4f}\n'
                  'Batch ~ BG Count: {:.4f}\n'
                  'Batch ~ NEG Count: {:.4f}\n'
                  .format(self.current_epoch, self.max_epochs,
                          index + 1, len(dataloader),
                          train_loss / (index + 1),
                          train_fg_loss / (index + 1),
                          train_bg_loss / (index + 1),
                          train_neg_loss / (index + 1),
                          n_fg_affinities ,
                          n_bg_affinities ,
                          n_neg_affinities
                          ))
            pass

        self.train_history["loss"].append(train_loss / len(dataloader))
        self.train_history["fg_loss"].append(train_fg_loss / len(dataloader))
        self.train_history["bg_loss"].append(train_bg_loss / len(dataloader))
        self.train_history["neg_loss"].append(train_neg_loss / len(dataloader))
        return

    def val_epoch(self, dataloader):

        val_loss = 0
        val_fg_loss = 0
        val_bg_loss = 0
        val_neg_loss = 0
        self.eval()
        with torch.no_grad():

            for index, data in enumerate(dataloader):

                img = data[1]
                labels = data[2]

                aff_pred = self(img)


                BG_labels = labels[0]
                FG_labels = labels[1]
                NEG_labels = labels[2]

                n_bg_affinities = torch.sum(BG_labels) + 1e-6
                n_fg_affinities = torch.sum(FG_labels) + 1e-6
                n_neg_affinities = torch.sum(NEG_labels) + 1e-6

                ## loss according to Eq. 7,8,9,10
                ## BG loss
                fg_loss = torch.sum(-FG_labels * torch.log(aff_pred+1e-6)) / n_fg_affinities
                bg_loss = torch.sum(-BG_labels * torch.log(aff_pred+1e-6)) / n_bg_affinities
                neg_loss = torch.sum(-NEG_labels * torch.log(1 - aff_pred + 1e-6)) / n_neg_affinities


                # explainability_cue, preds = self.extract_LRP(img)
                loss = fg_loss/4 + bg_loss/4 + neg_loss/2
                ##

                ### adding batch loss into the overall loss
                val_loss += loss.item()
                val_fg_loss += fg_loss.item()
                val_bg_loss += bg_loss.item()
                val_neg_loss += neg_loss.item()

                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Overall Loss: {:.4f}\n'
                      'Batch ~ FG Loss: {:.4f}\n'
                      'Batch ~ BG Loss: {:.4f}\n'
                      'Batch ~ NEG Loss: {:.4f}\n'
                      .format(self.current_epoch, self.max_epochs,
                              index + 1, len(dataloader),
                              val_loss / (index + 1),
                              val_fg_loss / (index + 1),
                              val_bg_loss / (index + 1),
                              val_neg_loss / (index + 1)))

            self.val_history["loss"].append(val_loss / len(dataloader))
            self.val_history["fg_loss"].append(val_fg_loss / len(dataloader))
            self.val_history["bg_loss"].append(val_bg_loss / len(dataloader))
            self.val_history["neg_loss"].append(val_neg_loss / len(dataloader))
            return


    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name + "/loss.png")
        plt.close()

        ## Plotting FG loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("FG-Loss")
        plt.title("FG - Loss Graph")

        plt.plot(np.arange(len(self.train_history["fg_loss"])), self.train_history["fg_loss"], label="train")
        plt.plot(np.arange(len(self.val_history["fg_loss"])), self.val_history["fg_loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name + "/fg.png")
        plt.close()


        ## Plotting BG loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Bg-Loss")
        plt.title("Bg-Loss Graph")

        plt.plot(np.arange(len(self.train_history["bg_loss"])), self.train_history["bg_loss"], label="train")
        plt.plot(np.arange(len(self.val_history["bg_loss"])), self.val_history["bg_loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name + "/bg.png")
        plt.close()

        ## Plotting NG loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Neg-Loss")
        plt.title("Neg-Loss Graph")

        plt.plot(np.arange(len(self.train_history["neg_loss"])), self.train_history["neg_loss"], label="train")
        plt.plot(np.arange(len(self.val_history["neg_loss"])), self.val_history["neg_loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name + "/neg.png")
        plt.close()




        return

    def load_pretrained(self, weights_path):

        ## loading weights
        weights_dict = torch.load(weights_path)

        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}

        no_pretrained_dict = {k: v for k, v in model_dict.items() if
                           not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


#########################################################################
def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(flatten)
            self.flatten = nn.Identity()
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


class MaxPool2dSame(nn.MaxPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D max pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        super(MaxPool2dSame, self).__init__(kernel_size, stride, (0, 0), dilation, ceil_mode)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride, value=-float('inf'))
        return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)


def _create_pool(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten_in_pool)
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop('padding', '')
    padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, **kwargs)
    if is_dynamic:
        if pool_type == 'avg':
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == 'max':
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'
    else:
        if pool_type == 'avg':
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == 'max':
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'

def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

from typing import Any, Callable, Optional, Tuple, List

def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic

def is_stem_deep(stem_type):
    return any([s in stem_type for s in ('deep', 'tiered')])

class GroupNormAct(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(GroupNormAct, self).__init__(num_groups, num_channels, eps=eps, affine=affine)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.act(x)
        return x

class DownsampleAvg(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None,
            preact=True, conv_layer=None, norm_layer=None):
        """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(self.pool(x)))

class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.
    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=False, eps=1e-6):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(
            self, in_chs, out_chs=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1,
            act_layer=None, conv_layer=None, norm_layer=None, proj_layer=None, drop_path_rate=0.):
        super().__init__()
        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, preact=True,
                conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.norm1 = norm_layer(in_chs)
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm2 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm3 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x_preact = self.norm1(x)

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x_preact)

        # residual branch
        x = self.conv1(x_preact)
        x = self.conv2(self.norm2(x))
        x = self.conv3(self.norm3(x))
        x = self.drop_path(x)
        return x + shortcut

# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)
# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.
    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding='SAME',
            dilation=1, groups=1, bias=False, eps=1e-6):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.same_pad = is_dynamic
        self.eps = eps

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def create_resnetv2_stem(
        in_chs, out_chs=64, stem_type='', preact=True,
        conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32)):
    stem = OrderedDict()
    assert stem_type in ('', 'fixed', 'same', 'deep', 'deep_fixed', 'deep_same', 'tiered')

    # NOTE conv padding mode can be changed by overriding the conv_layer def
    if is_stem_deep(stem_type):
        # A 3 deep 3x3  conv stack as in ResNet V1D models
        if 'tiered' in stem_type:
            stem_chs = (3 * out_chs // 8, out_chs // 2)  # 'T' resnets in resnet.py
        else:
            stem_chs = (out_chs // 2, out_chs // 2)  # 'D' ResNets
        stem['conv1'] = conv_layer(in_chs, stem_chs[0], kernel_size=3, stride=2)
        stem['norm1'] = norm_layer(stem_chs[0])
        stem['conv2'] = conv_layer(stem_chs[0], stem_chs[1], kernel_size=3, stride=1)
        stem['norm2'] = norm_layer(stem_chs[1])
        stem['conv3'] = conv_layer(stem_chs[1], out_chs, kernel_size=3, stride=1)
        if not preact:
            stem['norm3'] = norm_layer(out_chs)
    else:
        # The usual 7x7 stem conv
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)
        if not preact:
            stem['norm'] = norm_layer(out_chs)

    if 'fixed' in stem_type:
        # 'fixed' SAME padding approximation that is used in BiT models
        stem['pad'] = nn.ConstantPad2d(1, 0.)
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    elif 'same' in stem_type:
        # full, input size based 'SAME' padding, used in ViT Hybrid model
        stem['pool'] = create_pool2d('max', kernel_size=3, stride=2, padding='same')
    else:
        # the usual PyTorch symmetric padding
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    return nn.Sequential(stem)

def make_div(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        # NOTE: using my Linear wrapper that fixes AMP + torchscript casting issue
        fc = Linear(num_features, num_classes, bias=True)
    return fc
class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0., use_conv=False):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(in_chs, num_classes, pool_type, use_conv=use_conv)
        self.fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def forward(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        x = self.flatten(x)
        return x

class Bottleneck(nn.Module):
    """Non Pre-activation bottleneck block, equiv to V1.5/V1b Bottleneck. Used for ViT.
    """
    def __init__(
            self, in_chs, out_chs=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1,
            act_layer=None, conv_layer=None, norm_layer=None, proj_layer=None, drop_path_rate=0.):
        super().__init__()
        first_dilation = first_dilation or dilation
        act_layer = act_layer or nn.ReLU
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs, out_chs, stride=stride, dilation=dilation, preact=False,
                conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm1 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm2 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.norm3 = norm_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.act3 = act_layer(inplace=True)

    def forward(self, x):
        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        # residual
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.drop_path(x)
        x = self.act3(x + shortcut)
        return x

class DownsampleConv(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None, preact=True,
            conv_layer=None, norm_layer=None):
        super(DownsampleConv, self).__init__()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=stride)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(x))

class ResNetStage(nn.Module):
    """ResNet Stage."""
    def __init__(self, in_chs, out_chs, stride, dilation, depth, bottle_ratio=0.25, groups=1,
                 avg_down=False, block_dpr=None, block_fn=PreActBottleneck,
                 act_layer=None, conv_layer=None, norm_layer=None, **block_kwargs):
        super(ResNetStage, self).__init__()
        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer)
        proj_layer = DownsampleAvg if avg_down else DownsampleConv
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_idx in range(depth):
            drop_path_rate = block_dpr[block_idx] if block_dpr else 0.
            stride = stride if block_idx == 0 else 1
            self.blocks.add_module(str(block_idx), block_fn(
                prev_chs, out_chs, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups,
                first_dilation=first_dilation, proj_layer=proj_layer, drop_path_rate=drop_path_rate,
                **layer_kwargs, **block_kwargs))
            prev_chs = out_chs
            first_dilation = dilation
            proj_layer = None

    def forward(self, x):
        x = self.blocks(x)
        return x

def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def _init_weights(module: nn.Module, name: str = '', zero_init_last=True):
    if isinstance(module, nn.Linear) or ('head.fc' in name and isinstance(module, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif zero_init_last and hasattr(module, 'zero_init_last'):
        module.zero_init_last()

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    """

    def __init__(
            self, layers, channels=(256, 512, 1024, 2048),
            num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
            width_factor=1, stem_chs=64, stem_type='', avg_down=False, preact=True,
            act_layer=nn.ReLU, conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32),
            drop_rate=0., drop_path_rate=0., zero_init_last=False):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        wf = width_factor

        self.feature_info = []
        stem_chs = make_div(stem_chs * wf)
        self.stem = create_resnetv2_stem(
            in_chans, stem_chs, stem_type, preact, conv_layer=conv_layer, norm_layer=norm_layer)
        stem_feat = ('stem.conv3' if is_stem_deep(stem_type) else 'stem.conv') if preact else 'stem.norm'
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))

        prev_chs = stem_chs
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = PreActBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_div(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = ResNetStage(
                prev_chs, out_chs, stride=stride, dilation=dilation, depth=d, avg_down=avg_down,
                act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer, block_dpr=bdpr, block_fn=block_fn)
            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{stage_idx}')]
            self.stages.add_module(str(stage_idx), stage)

        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else nn.Identity()
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

        self.init_weights(zero_init_last=zero_init_last)


    def init_weights(self, zero_init_last=True):
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        _load_weights(self, checkpoint_path, prefix)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
