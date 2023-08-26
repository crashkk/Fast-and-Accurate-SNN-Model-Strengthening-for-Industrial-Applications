from industry_dataset.elpv_dataset.utils.elpv_reader import load_dataset
#process elpv data and return dataset objects
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torchvision.transforms import *
import torchvision
import torch
import numpy as np
from PIL import Image
from utils import *
import random
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler

def elpv_dataset_make():
    images,proba,types=load_dataset()#2624 in total
    class_list={#four classes
        0:0,
        1/3:1,
        2/3:2,
        1:3,
    }
    label=[]
    for p in proba:
        label.append(class_list[p])
    label=np.array(label)
    return images,label

class elpv(Dataset):
    def __init__(self,train=True,forget_mode=False,generator=elpv_dataset_make):
        self.forget_mode=forget_mode
        if args.regenerate_elpv_or_not=='Y':
            X,y=generator()
            random.seed(args.random_seed_for_elpv_traintestsplit)
            pos=list(i for i in range(len(y)))
            random.shuffle(pos)
            split_size=0.8#train test split
            X_train=X[pos[:int(split_size*len(y))]]
            y_train=y[pos[:int(split_size*len(y))]]
            X_test=X[pos[int(split_size*len(y)):]]
            y_test=y[pos[int(split_size*len(y)):]]
            np.savez('snn_forget_industry/industry_dataset/elpv_dataset/pregenerate_data/train.npz',data1=X_train,data2=y_train)
            np.savez('snn_forget_industry/industry_dataset/elpv_dataset/pregenerate_data/test.npz',data1=X_test,data2=y_test)
            if train==True:
                self.X=X_train
                self.y=y_train
            elif train==False:
                self.X=X_test
                self.y=y_test
        
        elif args.regenerate_NEU_64_or_not=='N':#read train and test data
            if train==True:
                data=np.load('snn_forget_industry/industry_dataset/elpv_dataset/pregenerate_data/train.npz')
            elif train==False:
                data=np.load('snn_forget_industry/industry_dataset/elpv_dataset/pregenerate_data/test.npz')
            self.X=data['data1']
            self.y=data['data2']

        if (train==True and not self.forget_mode) or args.retraining:
            transform=Compose([#300*300 to 128*128
                CenterCrop(265),
                transforms.RandomApply([
                    GaussianBlur(kernel_size=11,sigma=2),#gaussian blur
                ], p=0.5),
                #transforms.ToTensor(),
               # transforms.Normalize((0.5,), (0.5,)),
                #transforms.Lambda(lambda x: x.mul(255).to(torch.uint8)),
                #transforms.ToPILImage(),
                #torchvision.transforms.functional.equalize,
                
                
                Resize(64),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                #RandomRotation(degrees=180),
                transforms.ToTensor(),
                Normalize((0.5),(0.5))
            ])
        
        elif train==False or self.forget_mode:#dataset that is used for model forget don't need to be transformed
            transform=Compose([
                CenterCrop(265),
                #GaussianBlur(kernel_size=265,sigma=2),#gaussian blur
                #transforms.ToTensor(),
                #transforms.Normalize((0.5,), (0.5,)),
                #transforms.Lambda(lambda x: x.mul(255).to(torch.uint8)),
                #transforms.ToPILImage(),
                #torchvision.transforms.functional.equalize,
                
                
                Resize(64),
                transforms.ToTensor(),
                Normalize((0.5),(0.5))
            ])

        self.transform=transform
        self.train=train
        #self.resampler=RandomUnderSampler(random_state=678)
        #self.resampler=SMOTE(random_state=678)
        self.resampler=RandomOverSampler(random_state=1444)

    def __getitem__(self, index):
        x=self.X[index]
        y=self.y[index]

        if self.train==False:
            x=x.reshape(300,300)

        if self.transform:
            x=Image.fromarray(x)
            x=self.transform(x)

        return x,y
    
    def __len__(self):
        return len(self.X)
    
    def resample(self,sample=True):#oversample to compensate imbalance of classes of data
        X_slice,y_slice=self.X,self.y
        if sample==False:
            self.X=X_slice.reshape(len(X_slice),300,300)
            self.y=y_slice

        X_slice=X_slice.reshape(len(X_slice),-1)
        X_resampled,y_resampled=self.resampler.fit_resample(X_slice,y_slice)
        X_resampled=X_resampled.reshape(len(X_resampled),300,300)
        self.X=X_resampled
        self.y=y_resampled

if __name__ == '__main__':
    train_dataset=elpv(train=True)
    train_dataset.resample()
    
    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=5,shuffle=True)
    for idx,(data,label) in enumerate(train_dataloader):
        print(data.shape,label)
    