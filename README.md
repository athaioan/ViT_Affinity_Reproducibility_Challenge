# [RE] Weakly-Supervised Semantic Segmentation via Transformer Explainability (Reproduction of Transformer Interpretability Beyond Attention Visualization [CVPR 2021])

## Openreview
You can find the report at http://dx.doi.org/10.5281/zenodo.6574631

## Requirements
You can download the pretrained weights we used and those we trained at https://drive.google.com/drive/folders/1lCjPjO3_BUnk-utY9-taCTQKoq2Pm43s


### Create and activate a virtual environment

Creation and activation of virtual environments in Linux systems is done by executing the command venv:
```bash
python3 -m venv /path/to/new/virtual/environment
. /path/to/new/virtual/environment/bin/activate
```

When using Anaconda Creation and activation of virtual environments in Linux systems is done by executing the following command:
```bash
conda create --name /path/to/new/virtual/environment python=3.8 numpy
source activate /path/to/new/virtual/environment
```

### Install dependencies
Installation of the required libraries to run our code can be achieved by the following command:
```bash
pip install -r requirements.txt
```

### Download the different datasets
Downloading ImageNet 2012 validation set, ImageNet segmentation dataset and PascalVOC 2012, can be achieved by running the following commands respectively:
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

## Experiments
To reproduce the results of Tables 2 and 3 referring to the ViT-base interpretability run demo.py

To reproduce the results of Table 4 referring to the ViT-Hybrid run demo_PascalVocHybrid.py 

To reproduce the results of Table 4 referring to the Affinity ViT-Hybrid run demo_PascalVocHybrid_Affinity.py 

## Citation
To cite our work please use:
```@inproceedings{
athanasiadis2022weaklysupervised,
title={Weakly-Supervised Semantic Segmentation via Transformer Explainability},
author={Ioannis Athanasiadis and Georgios Moschovis and Alexander Tuoma},
booktitle={ML Reproducibility Challenge 2021 (Fall Edition)},
year={2022},
url={https://openreview.net/forum?id=rcEDhGX3AY}
}
```

