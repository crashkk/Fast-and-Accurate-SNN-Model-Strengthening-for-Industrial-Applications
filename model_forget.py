from __future__ import print_function
from ast import arg
import gc
from tkinter import Variable
from math import log
import numpy as np
from utils import *
from sample_forget_dataset import dataprocess
import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
from torchvision import datasets, transforms
import time
from SNN_model.shallow_spiking_model2 import *
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

use_cuda=args.cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

writer=SummaryWriter('./forget_experiment_log')

def KL_divergence(OT,WT):#定义KL散度损失函数
    logp_wt=F.log_softmax(WT,dim=-1)
    p_ot=F.softmax(OT, dim=-1)
    kl= F.kl_div(logp_wt, p_ot, reduction='sum')
    return kl

def entropycal(output):#计算信息熵。输入是tensor类型,返回一个信息熵列表
    n=output.shape[0]
    ent=[]#记录每个样本output的信息熵
    for i in range(n):
        ent_log=0
        for j in range(output.shape[1]):
            if output[i,j]==0:
                continue
            prob=output[i,j].item()/torch.sum(output[i,:]).item()
            ent_log+=-(prob*log(prob))
        ent.append(ent_log)
    return ent

def model_forget(args, forget_model, target_model,optimizer, epoch,third_party_dataloader,forgetclass_dataloader,nonforgetclass_dataloader,subset_or_class):
    lamb=args.lambd#balance loss function cifar10:0.01,mnist:0.7
    target_model.eval()
    breakpointt=False

    
    for batch_idx, (forget_data, tag1) in enumerate(forgetclass_dataloader):
        forget_model.eval()
        loss2=0
        with torch.no_grad():
            for data, target in nonforgetclass_dataloader:
                data, target = data.to(device), target.to(device)
                
                output = forget_model(data)
                loss2 += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
        loss2 = loss2*(1-lamb)/len(nonforgetclass_dataloader.dataset)
        forget_model.train()
        forget_data, tag1 = forget_data.to(device), tag1.to(device)

        if subset_or_class==True:#the third-party data used in iterations is from test dataset
            bi=0
            for third_data,_ in third_party_dataloader:
                third_data= third_data.to(device)
                if bi==batch_idx:
                    break
                else:
                    bi=bi+1
                    continue
        else:#the third-party data used in iterations is from random noise dataset thus received data only
            for bi,(third_data) in enumerate(third_party_dataloader):
                third_data = third_data.float()
                third_data= third_data.to(device)
                if bi==batch_idx:
                    break
                else:
                    continue
        
        ot = target_model(third_data)#第三方数据输入SNNS给出目标值
        wt = forget_model(forget_data)#待遗忘的数据输入SNNS给出预测值
        optimizer.zero_grad() 
        loss= KL_divergence(ot,wt)*lamb+loss2
        loss.backward()
        optimizer.step()

        '''
        #this block is for ending the iterations
        if loss2/(1-lamb)>=1.65:#1.55 this mechanism is for nmnist and mnist
            print('forget Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
            print(loss2/(1-lamb))
            breakpointt=True
            break
        '''
        _,breakpointt=efficiency_test(args, forget_model,forgetclass_dataloader , epoch,subset_or_class)
        if breakpointt==True:
            break

        if (batch_idx+1) % (args.log_interval+1) == 0:
            print('forget Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
            #print(loss2/(1-lamb))
        
            writer.add_scalar('Forget Train Loss /batchidx', loss, batch_idx + args.forget_batch * epoch)

    return forget_model,breakpointt

def efficiency_test(args, forget_model, forget_dataset_loader, epoch,subset_or_class):#用遗忘后的模型预测遗忘样本
    forget_model.eval()
    breakpointt=False
    test_loss = 0
    correct = 0
    entropy=[]
    with torch.no_grad():
        for data, target in forget_dataset_loader:
            data, target = data.to(device), target.to(device)
            
            output = forget_model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            entropy.append(entropycal(output))
            correct += pred.eq(target.view_as(pred)).sum().item()#计算每个batch的平均预测率,期望值应该是10%
    corr_rate=100. * correct / len(forget_dataset_loader.dataset)#存储正确率
    writer.add_scalar('Test epoch', int(epoch), epoch)
    writer.add_scalar('Test Acc /epoch', corr_rate, epoch)
    for i, (name, param) in enumerate(forget_model.named_parameters()):
        if '_s' in name:
            writer.add_histogram(name, param, epoch)

    print('\n forget_efficiency_test: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(forget_dataset_loader.dataset),corr_rate))

    if subset_or_class==False:
        if corr_rate<=8:
            breakpointt=True
    return entropy,breakpointt

def SPMU(target_model_log_path,forget_model_log_path):
    gc.collect()
    torch.cuda.empty_cache()
    #train_loader,non_forget_dataset,non_forget_dataset_loader,forget_dataset_loader,test_loader,third_dataset,total_train_dataset=dataprocess()
    #forgetclass_dataloader,nonforgetclass_dataloader,third_dataset=dataprocess2()

    subset_or_class=args.subset_or_class#False denotes class forgotten while True denotes subset forgotten

    forgetclass_dataloader,nonforgetclass_dataloader,total_train_dataset,test_loader,third=dataprocess(subset=subset_or_class)
    
    forget_model = SCNN().to(device)
    #model = nn.DataParallel(model,device_ids=[0,1])

    target_model = SCNN().to(device)
    #target_model = nn.DataParallel(target_model,device_ids=[0,1])
    optimizer = optim.Adam(forget_model.parameters(), lr=args.lr_forget)
    
    target_model.load_state_dict(torch.load(target_model_log_path))#设置目标模型和待遗忘模型初始值一样
    forget_model.load_state_dict(torch.load(target_model_log_path))
    print('pretrained-target-model loaded successfully!')
    #bpt=False
    """
    这一模块进行SNNS的遗忘
    """
    efficiency_test(args, forget_model, forgetclass_dataloader, 0,subset_or_class)
    start_time=time.time()
    for epoch in range(1,args.model_forget_epochs + 1 ):
        forget_model,bpt=model_forget(args, forget_model, target_model,optimizer, epoch,third,forgetclass_dataloader,nonforgetclass_dataloader,subset_or_class)
        ent_log,bpt=efficiency_test(args, forget_model, forgetclass_dataloader, epoch,subset_or_class)#暂时用直接预测遗忘样本的方式验证，观察预测结果是否在10%左右
        if bpt==True:
            break
    end_time=time.time()
    total_time=float(end_time-start_time)
    #print('脉冲网络每一层的总放电频数：{}'.format(forget_model.spiking_frequecy))
    print('SNNS遗忘耗时为：{:.4f}min'.format(total_time/60))
    writer.close()

    if (args.save_forget_model):
        torch.save(forget_model.state_dict(),  forget_model_log_path)#存储模型权重参数
        torch.save(forget_model, forget_model_log_path+'h')#存储模型结构

    '''
    ent_log=np.array(ent_log)
    ent_log=np.reshape(ent_log,(1,ent_log.shape[0]*ent_log.shape[1]))
    plt.figure()
    plt.hist(np.squeeze(ent_log), bins=20, color="aliceblue")
    plt.savefig('hist')
    '''
