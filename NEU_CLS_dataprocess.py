#process NEU_CLS data and return dataset objects
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torchvision.transforms import *
import torch
import numpy as np
from PIL import Image
from industry_dataset.read_bmpfile import optimal_mean_std
from utils import *

#every figure 200*200
#six defect types:['Pa', 'Sc', 'Cr', 'In', 'RS', 'PS']
#and mark them from 0 to 5 respectively
classes=['Pa', 'Sc', 'Cr', 'In', 'RS', 'PS']
def abstract_label(file_path,classes=classes):
    label=[]
    dirr=os.listdir(file_path)
    for filename in dirr:
        pick=filename[:2]
        label.append(classes.index(pick))
        q=filename
    return label

def read_bmp(image):
    pixels=list(image)

class NEU_CLS(Dataset):

    def __init__(self,train=True,forget_mode=False,network='resnet',image_dir='snn_forget_industry/industry_dataset/NEU_CLS/',transform=None,abstarct_label=abstract_label):
        self.forget_mode=forget_mode
        if train==True:
            self.img_dir=image_dir+'Train'
        else:
            self.img_dir=image_dir+'Test'
        
        self.img_labels=abstract_label(self.img_dir)
        #op_mean,op_std=optimal_mean_std(self.img_dir)

        if network=='SCNN':
            if (train==True and not self.forget_mode) or args.retraining:
                transform=Compose([
                    transforms.ToTensor(),
                    Resize(128),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=10),
                    transforms.Normalize([0.5], [0.5])])
            elif train==False or self.forget_mode:
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(128),
                    transforms.Normalize([0.5], [0.5])])
                                                                    
        self.transform=transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        files=os.listdir(self.img_dir)
        current_filename=files[index]
        img_path=os.path.join(self.img_dir,current_filename)
        image=Image.open(img_path)
        image = image.convert('L') # 转换为灰度图
        image = np.array(image, dtype=np.uint8) # 转换为numpy数组，数据类型为uint8
        #resize(200*200)->(64*64)
        #image=image.resize((64,64),Image.ADAPTIVE)
        image=np.array(image)
        label=self.img_labels[index]

        if self.transform:
            image=self.transform(image)
        return image,label
if __name__=='__main__':
    #label=abstract_label('snn_forget_industry/industry_dataset/NEU_CLS/')
    #print(label)
    
    total_dataset=NEU_CLS(train=False)
    total_data=torch.utils.data.DataLoader(total_dataset,batch_size=5,shuffle=False)
    for idx,(image,label) in enumerate(total_data):
        print(image,label)
    
