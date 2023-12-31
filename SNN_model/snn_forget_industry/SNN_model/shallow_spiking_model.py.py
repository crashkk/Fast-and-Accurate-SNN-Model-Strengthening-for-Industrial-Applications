import torch#CNN based structure
import torch.nn as nn
import torch.nn.functional as F
from ast import arg
import sys
sys.path.append('/home/zdm/snn_forget_industry/utils.py')
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

probs = 0.5#drop out rate
tw=args.time_steps_CNN
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = args.numclass
batch_size  = args.batch_size#100
inputsize=args.inputsize

class ActFun(torch.autograd.Function):
    # Define approximate firing function
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # au function
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()
act_fun = ActFun.apply
# membrane potential update
def mem_update(conv, x, mem, spike):
    mem = mem * decay * (1. - spike) + conv(x)
    spike = act_fun(mem)
    return mem, spike


# cnn layer :(in_plane,out_plane, stride,padding, kernel_size)
cfg_cnn = [(1, 128, 1, 1, 3),

           (128, 256, 1, 1, 3),

           (256, 512, 1, 1, 3),

           (512, 1024, 1, 1, 3),

           (1024, 512, 1, 1, 3),
           ]

cfg_kernel = [int(inputsize), int(inputsize), int(inputsize/2), int(inputsize/(2**2)), int(inputsize/(2**2)), int(inputsize/(2**2))]
# fc layer
cfg_fc = [1024, 512, int(num_classes*10)]

# voting matrix
weights = torch.zeros(cfg_fc[-1], num_classes, device=device,requires_grad = False)  # cfg_fc[-1]
vote_num = cfg_fc[-1] // num_classes
for i in range(cfg_fc[-1]):
    weights.data[i][i // vote_num] = num_classes / cfg_fc[-1]

def assign_optimizer(model, lrs=1e-3):
    rate = 1e-1
    fc1_params = list(map(id, model.fc1.parameters()))
    fc2_params = list(map(id, model.fc2.parameters()))
    fc3_params = list(map(id, model.fc3.parameters()))
    base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params, model.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': model.fc1.parameters(), 'lr': lrs * rate},
        {'params': model.fc2.parameters(), 'lr': lrs * rate},
        {'params': model.fc3.parameters(), 'lr': lrs * rate}, ]
        , lr=lrs, momentum=0.9)
    print('successfully reset lr')
    return optimizer


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


class SCNN(nn.Module):

    def __init__(self, num_classes=num_classes):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[4]
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])

    def forward(self, input,simulation_required=False):
        batch_size=list(input.size())[0]
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_mem = c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device)
        c5_mem = c5_spike = torch.zeros(batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)
        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, cfg_fc[2], device=device)

        if simulation_required==True:
            tw=args.simulation_time
        else:
            tw=args.time_steps_CNN

        for step in range(tw):

        
            #x = input > torch.rand(input.size(), device=device)
            x = input
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)

            c2_mem, c2_spike = mem_update(self.conv2, F.dropout(c1_spike, p=probs, training=self.training), c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 2)
            x = F.dropout(x, p=probs, training=self.training)

            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike)
            
            x = F.avg_pool2d(c3_spike, 2)
            
            x = F.dropout(x, p=probs, training=self.training)
            
            c4_mem, c4_spike = mem_update(self.conv4, x, c4_mem, c4_spike)
            
            x = F.dropout(c4_spike, p=probs, training=self.training)
            c5_mem, c5_spike = mem_update(self.conv5, x, c5_mem, c5_spike)
            x = c5_spike.view(batch_size, -1)
            
            h1_mem, h1_spike = mem_update(self.fc1, F.dropout(x, p=probs, training=self.training), h1_mem, h1_spike)
            
            h2_mem, h2_spike = mem_update(self.fc2, F.dropout(h1_spike, p=probs, training=self.training), h2_mem,h2_spike)

            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike
  
        outputs = h3_sumspike / tw
        
        outputs = outputs.mm(weights)

        return outputs
