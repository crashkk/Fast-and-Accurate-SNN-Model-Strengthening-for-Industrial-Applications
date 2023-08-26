from __future__ import print_function
from ast import arg
from weight_cal import weights as wc
from pickletools import optimize
import numpy as np
#from spdata_dataset_devide import dataprocess
import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from utils import *
from torchvision import datasets, transforms
#from resnet_model_mnist import *
from SNN_model.shallow_spiking_model2 import *
from NEU_CLS_dataprocess import NEU_CLS
from NEU_CLS_64_dataprocess import NEU_CLS_64
from elpv_dataprocess import elpv
from tensorboardX import SummaryWriter

use_cuda=args.cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

steps=args.time_steps_CNN
criterion = torch.nn.CrossEntropyLoss(weight=wc().to(device))

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr_CNN * (0.1 ** (epoch // 35))#35 default
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def confusion_matrix(ground_truth,predict,size=args.numclass):
    matrix=torch.zeros((size,size))
    for id1,id2 in zip(ground_truth,predict):
        matrix[id1,id2]+=1
    print('confusion matrix:{}'.format(matrix))
    acc_classes=matrix.diag()/matrix.sum(1)
    print('accuracy of each class',acc_classes)

def model_train(args, target_model, device, optimizer, epoch, writer ,total_train_dataloader,scheduler):#训练包含被遗忘样本的模型
    adjust_learning_rate(args,optimizer,epoch)
    #scheduler.step()
    target_model.train()
    for batch_idx1, (data, target) in enumerate(total_train_dataloader):
        #target= torch.topk(target, 1)[1].squeeze(1)
        data, target = data.to(device), target.to(device)

        # necessary for general dataset: broadcast input
        #data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape)) 
        #data = data.permute(1, 2, 3, 4, 0)

        if epoch==0 and batch_idx1==0:
            output = target_model(data,simulation_required=True)
        else:
            output = target_model(data)
        loss = F.cross_entropy(output, target)
        #loss=criterion(output, target)
        #loss =F.mse_loss(output,target)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        if batch_idx1 % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlearning_rate:{:.7f}'.format(
                epoch, batch_idx1 * len(data / steps), len(total_train_dataloader.dataset),
                    100. * batch_idx1 * len(data / steps)/ len(total_train_dataloader.dataset), loss.item(),optimizer.state_dict()['param_groups'][0]['lr']))
            writer.add_scalar('Train Loss /batchidx', loss, batch_idx1 + len(total_train_dataloader) * epoch)

def test(args, target_model, device, test_loader, epoch, writer):
    target_model.eval()    
    test_loss = 0
    correct = 0
    isEval = False
    label_list=[]
    pred_list=[]

    with torch.no_grad():
        for data, target in test_loader:
            #target= torch.topk(target, 1)[1].squeeze(1)
            data, target = data.to(device), target.to(device)

            if len(label_list)==0:
                label_list=target
            else:
                label_list=torch.cat([label_list,target],dim=0)

            #data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            #data = data.permute(1, 2, 3, 4, 0)
            output = target_model(data)
            #test_loss +=criterion(output, target).item()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if len(pred_list)==0:
                pred_list=pred
            else:
                pred_list=torch.cat([pred_list,pred],dim=0)


    test_loss /= len(test_loader.dataset)

    writer.add_scalar('Test Loss /epoch', test_loss, epoch)   
    writer.add_scalar('Test Acc /epoch', 100. * correct / len(test_loader.dataset), epoch)
    for i, (name, param) in enumerate(target_model.named_parameters()):
        if '_s' in name:
            writer.add_histogram(name, param,epoch)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    confusion_matrix(label_list,pred_list)

def train_SCNN(dataset='NEU_CLS_64'):
    
    torch.manual_seed(1348)
    
    if dataset=='NEU_CLS':
        train_dataloader=torch.utils.data.DataLoader(NEU_CLS(train=True,network='SCNN'),batch_size=args.batch_size, shuffle=True,**kwargs)
        test_dataloader=torch.utils.data.DataLoader(NEU_CLS(train=False,network='SCNN'),batch_size=args.batch_size, shuffle=False,**kwargs)
    elif dataset=='NEU_CLS_64':
        train_dataset=NEU_CLS_64(train=True)
        train_dataset.resample()
        train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True,**kwargs)
        
        test_dataset=NEU_CLS_64(train=False)#test dataset don't need to resample
        test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False,**kwargs)
    elif dataset=='elpv':
        train_dataset=elpv(train=True)
        train_dataset.resample()
        train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True,**kwargs)
        
        test_dataset=elpv(train=False)#test dataset don't need to resample
        test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False,**kwargs)

    writer=SummaryWriter('./target_model_experiment_log')
    
    target_model= SCNN().to(device)#this for shallow snn
    #target_model = nn.DataParallel(target_model,device_ids=[1,2])  #this for parallel training(not suitable for cifar or dvs-cifar)
    optimizer = optim.Adam(target_model.parameters(), lr=args.lr_CNN)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 10, verbose = False, threshold = 0.0001, threshold_mode = 'rel', cooldown = 0, min_lr = 0, eps = 1e-08)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)

    target_model_log_path='snn_forget_industry/experiment_log/target_model/'+'SCNN_'+dataset+'_target_model.pt'#经过训练不含被遗忘数据的目标模型参数
    start_time=time.time()
    """
    这一模块进行SNNS的基础训练
    """
    if os.path.isfile(target_model_log_path) and not args.first_train:#target model train
        pretrained_target_model=torch.load(target_model_log_path)
        target_model.load_state_dict(pretrained_target_model)
        print('pretrained-target-model loaded successfully!')

        for epoch in range(args.target_train_epochs):
            model_train(args, target_model, device, optimizer,epoch, writer,train_dataloader,scheduler)#先对选定的总样本进行训练
            test(args, target_model, device, test_dataloader, epoch, writer)
            if epoch%args.break_point==0:
                torch.save(target_model.state_dict(),  'snn_forget_industry/experiment_log/target_model/'+'SCNN_'+dataset+'_target_model.pt')#存储模型权重参数
                torch.save(target_model, 'snn_forget_industry/experiment_log/target_model/'+'SCNN_'+dataset+'_target_model.pth')#存储模型结构

    
        if (args.save_pretrained_model):#对训练好的SNNS数据进行保存
            torch.save(target_model.state_dict(),  'snn_forget_industry/experiment_log/target_model/'+'SCNN_'+dataset+'_target_model.pt')#存储模型权重参数
            torch.save(target_model, 'snn_forget_industry/experiment_log/target_model/'+'SCNN_'+dataset+'_target_model.pth')#存储模型结构



    else:
        for epoch in range(args.target_train_epochs):
            model_train(args, target_model, device, optimizer, epoch, writer,train_dataloader,scheduler)#先对选定的总样本进行训练
            test(args, target_model, device, test_dataloader, epoch, writer)
            if epoch%args.break_point==0:
                torch.save(target_model.state_dict(),  'snn_forget_industry/experiment_log/target_model/'+'SCNN_'+dataset+'_target_model.pt')#存储模型权重参数
                torch.save(target_model, 'snn_forget_industry/experiment_log/target_model/'+'SCNN_'+dataset+'_target_model.pth')#存储模型结构

        if (args.save_pretrained_model):#对训练好的SNNS数据进行保存
            torch.save(target_model.state_dict(),  'snn_forget_industry/experiment_log/target_model/'+'SCNN_'+dataset+'_target_model.pt')#存储模型权重参数
            torch.save(target_model, 'snn_forget_industry/experiment_log/target_model/'+'SCNN_'+dataset+'_target_model.pth')#存储模型结构

    end_time=time.time()
    total_time=float(end_time-start_time)
    print('SNNS训练耗时为：{:.4f}min'.format(total_time/60))
    writer.close()

if __name__ == '__main__':
    ######train target model#########
    traindataset='NEU_CLS_64'
    train_SCNN(traindataset)#python snn_forget_industry/train_target_model_SCNN.py --numclass 9 --inputsize 32