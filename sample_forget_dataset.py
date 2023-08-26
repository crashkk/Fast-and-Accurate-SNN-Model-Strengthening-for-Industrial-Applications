from __future__ import print_function
from ast import arg
from ctypes.wintypes import tagRECT
from pickle import FALSE
from random import shuffle
import numpy as np
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from NEU_CLS_64_dataprocess import NEU_CLS_64
from NEU_CLS_dataprocess import NEU_CLS
from elpv_dataprocess import elpv
from noise_pic_generate import noise_img_intro

def dataprocess(subset=False,mem_inf=False):#
    torch.manual_seed(args.random_seed_for_forgetdata_split)

    use_cuda=args.cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}
    
    if args.dataset_type=='NEU_CLS':
        train_dataset=NEU_CLS(train=True,forget_mode=True,network='SCNN')
        test_dataset=NEU_CLS(train=False,network='SCNN')
    elif args.dataset_type=='NEU_CLS_64':
        train_dataset=NEU_CLS_64(train=True,forget_mode=True)
        unresampled_train_dataset=train_dataset#preserve unresampled dataset for subset forget
        train_dataset.resample()

        test_dataset=NEU_CLS_64(train=False)#test dataset don't need to resample
    elif args.dataset_type=='elpv':
        train_dataset=elpv(train=True,forget_mode=True)
        unresampled_train_dataset=train_dataset#preserve unresampled dataset for subset forget
        train_dataset.resample()

        test_dataset=elpv(train=False)#test dataset don't need to resample

    if subset==False:#class forget
        num_perclass=(train_dataset.__len__())/args.numclass
        third_dataloader=noise_img_intro(num_perclass)#random noise dataset
        point_class=args.forget_class
        point_class_list=[]
        for idx in range(train_dataset.__len__()):
            _,label=train_dataset.__getitem__(idx)#search point class index
            if args.dataset_type=='NEU_CLS_64' or args.dataset_type=='elpv':
                label=label.item()
            if label==point_class:
                point_class_list.append(idx)
        other_class_list=list(set([x for x in range(train_dataset.__len__())])-set(point_class_list))
        non_forget_dataset=torch.utils.data.Subset(train_dataset,other_class_list)
        forget_dataset=torch.utils.data.Subset(train_dataset,point_class_list)

        nonforget_dataloader=torch.utils.data.DataLoader(non_forget_dataset,batch_size=args.batch_size,shuffle=args.retraining,**kwargs)
        forget_dataloader=torch.utils.data.DataLoader(forget_dataset,batch_size=args.batch_size,shuffle=False,**kwargs)
        test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,**kwargs)
        print('forget dataset size:{},nonforget dataset size:{}'.format(len(forget_dataloader.dataset),len(nonforget_dataloader.dataset)))
        return forget_dataloader,nonforget_dataloader,train_dataset,test_loader,third_dataloader




if __name__== '__main__':
    dataprocess()
    print('data process done!')