from __future__ import print_function
from ast import arg
import numpy as np
from utils import *
from sample_forget_dataset import dataprocess
#from spdata_dataset_devide import dataprocess
import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from SNN_model.shallow_spiking_model2 import *
from tensorboardX import SummaryWriter
from result_analize.metrics_packs import confusions,F1_recall_precision

def nonforgotten_acc_test(args, forget_model, device, non_forget_dataset_loader):#用遗忘后的模型预测第三方数据、非遗忘数据，测试精度变化
    forget_model.eval()
    test_loss = 0
    correct = 0
    isEval = False
    with torch.no_grad():
        for data, target in non_forget_dataset_loader:
            #target= torch.topk(target, 1)[1].squeeze(1)#this is for nmnist target
            data, target = data.to(device), target.to(device)
            #data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            #data = data.permute(1, 2, 3, 4, 0)

            output = forget_model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(non_forget_dataset_loader.dataset)

    print('\nTest set in nonforget_dataset_loader: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(non_forget_dataset_loader.dataset),
        100. * correct / len(non_forget_dataset_loader.dataset)))


def third_acc_test(args, forget_model, device, third_dataset_loader):
    forget_model.eval()
    test_loss = 0
    correct = 0
    isEval = False

    target_list=[]
    pred_list=[]

    with torch.no_grad():
        for data, target in third_dataset_loader:
            #target= torch.topk(target, 1)[1].squeeze(1)#this is for nmnist target
            data, target = data.to(device), target.to(device)

            if len(target_list)==0:
                target_list=target
            else:
                target_list=torch.cat([target_list,target],dim=0)

            #data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            #data = data.permute(1, 2, 3, 4, 0)
            output = forget_model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if len(pred_list)==0:
                pred_list=pred
            else:
                pred_list=torch.cat([pred_list,pred],dim=0)

    test_loss /= len(third_dataset_loader.dataset)

    print('\nTest set in third_dataset: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(third_dataset_loader.dataset),
        100. * correct / len(third_dataset_loader.dataset)))
    


    confusions(np.array(target_list.cpu()),np.array(pred_list.cpu()))
    F1_recall_precision(np.array(target_list.cpu()),np.array(pred_list.cpu()))

def forgotten_acc_test(args, forget_model, device, forget_dataset_loader):#用遗忘后的模型预测第三方数据、非遗忘数据，测试精度变化
    forget_model.eval()
    test_loss = 0
    correct = 0
    isEval = False
    with torch.no_grad():
        for data, target in forget_dataset_loader:
            #target= torch.topk(target, 1)[1].squeeze(1)#this is for nmnist target                     
            data, target = data.to(device), target.to(device)
            #data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            #data = data.permute(1, 2, 3, 4, 0)
            output = forget_model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(forget_dataset_loader.dataset)

    print('\nTest set in forget_dataset_loader: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(forget_dataset_loader.dataset),
        100. * correct / len(forget_dataset_loader.dataset)))

def model_test(model_path):
    #train_loader,non_forget_dataset,non_forget_dataset_loader,forget_dataset_loader,test_loader,third_dataset,total_train_dataset=dataprocess(devide=False)
    forgetclass_dataloader,nonforgetclass_dataloader,total_train_dataset,test_loader,third_dataset=dataprocess()
    #forgetclass_dataloader,nonforgetclass_dataloader,total_train_dataset,test_loader,third_dataset=dataprocess()
    use_cuda=args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}
    
    third_dataset_loader=torch.utils.data.DataLoader(third_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
    print(len(forgetclass_dataloader.dataset))
    print(len(third_dataset_loader.dataset))
    
    forget_model = SCNN().to(device)
    forget_model.load_state_dict(torch.load(model_path),False)
    print('model loaded successfully!')
    
    forgotten_acc_test(args, forget_model, device, forgetclass_dataloader)
    nonforgotten_acc_test(args, forget_model, device,nonforgetclass_dataloader)
    third_acc_test(args, forget_model, device,test_loader)



