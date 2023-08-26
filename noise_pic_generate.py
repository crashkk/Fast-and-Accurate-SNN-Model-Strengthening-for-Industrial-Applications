import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
import numpy as np
from PIL import Image 
from torchvision.transforms import ToPILImage
import torch
from torchvision.io import read_image
from utils import *
from torchvision.transforms import *

class Mydataset(Dataset):
    def __init__(self,img_dir,dataset_type):#transform用compose方法包装
        super().__init__()
        self.img_dir=img_dir#照片文件所在的目录

        transform=Compose([
            ToTensor(),
            Normalize((0.5),(0.5)),
        ]
        )

        self.transform=transform
        self.dataset_type=dataset_type

    def __len__(self):
        return int(3000)
    
    def __getitem__(self, index):
        picname=str(int(index))+'.jpg'
        img_path=os.path.join(self.img_dir+picname)
        image=read_image(img_path)
        image=np.array(image)
        if self.dataset_type=='NEU_CLS':
            image=image.reshape(128,128)
        elif self.dataset_type=='NEU_CLS_64':
            image=image.reshape(64,64)
        if self.transform is not None:
            image=self.transform(image)
        return image

"""
def img_transform():#将普通图片转换为相应的数据集格式（cifar10,mnisit）
    img_dir='snns/dataset/new_data/'
    img_list=os.listdir(img_dir)

    sum_rgb=[]
    sum_img=[]
    k=k+1

    for img_name in img_list:
        img_path=os.path.join(img_dir,img_name)
        img=Image.open(img_path,'r')
        r,g,b=img.split()
        sum_rgb.append(np.array(r))
        sum_rgb.append(np.array(g))
        sum_rgb.append(np.array(b)) 
        sum_img.append(sum_rgb)
        sum_rgb=[]
        k=k+1
    
"""
def random_noise(nc,width,height):
    img = torch.rand(nc,width,height)
    img = ToPILImage()(img)
    return img

def noise_img_intro(forget_length):
    forget_length=int(forget_length)
    if args.dataset_type=='NEU_CLS' or args.dataset_type=='elpv':
        img_root_path='snn_forget_industry/industry_dataset/noise_dataset_128/'
    elif args.dataset_type=='NEU_CLS_64':
        img_root_path='snn_forget_industry/industry_dataset/noise_dataset/'
    noise_dataset=Mydataset(img_root_path,args.dataset_type)
    indices = torch.randperm(len(noise_dataset))[:forget_length]
    noise_dataset=torch.utils.data.Subset(noise_dataset,indices)
    noise_dataloader=torch.utils.data.DataLoader(noise_dataset,batch_size=args.batch_size,shuffle=False)
    return noise_dataloader

if __name__ == '__main__':
    data_exist=False#generate noise pictures
    if  not data_exist:
        for i in range(0,3000):
            picname=str(int(i))
            picname=picname+'.jpg'
            random_noise(1,128,128).save('snn_forget_industry/industry_dataset/noise_dataset_128/'+picname)#NEU-CLS:1*64*64
    noise_img_intro(1150)
    
