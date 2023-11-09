import numpy as np 
import matplotlib.pyplot as plt

# Torch
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

# Save/Load as mat.
from scipy.io import savemat, loadmat
import h5py
import mat73 # https://github.com/skjerns/mat7.3

# Progress Bar
from rich.progress import track
from rich.progress import Progress

# Other
import argparse
import multiprocessing as mp
import os
import scipy.signal as signal
import time

# User-defined
from mat2np_segment_all_subject import *
from dsp_preprocess import *
from dataset_parser import *
from models import *
from feature_extractor import *
from set_args import *
from train_test_process import *
from vit import *

set_seed(87)

# Parameter setup
args = get_args()
window_size, window_step, number_gesture, model_PATH, device = get_args_info(args)

# Dataset setup
train_loader, valid_loader, test_loader = train_test_split_DataLoader(\
                                        batch_size=args.batch_size, subject_list=args.subject_list, exercise_list= args.exercise_list, \
                                        fs=args.fs, window_size=window_size, window_step=window_step, num_channel=args.num_channel, \
                                        feat_extract=args.feat_extract, class_rest=args.class_rest)

if args.model_type == "ViT":
    # TNet
    num_patch = args.num_channel
    patch_size = window_size
    model = ViT(num_patch,patch_size).to(device)
else:
    model = eval(f"{args.model_type}(number_gesture=number_gesture, class_rest=args.class_rest)").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-6, verbose=True)
criterion = nn.CrossEntropyLoss()

# Model Training and Validation
if args.en_train:
    train_process(args,model,model_PATH,train_loader,valid_loader,device,optimizer,criterion, scheduler=scheduler)

# Model Testing
test_process(model,model_PATH,test_loader,device,criterion,args.load_model,args.model_type)