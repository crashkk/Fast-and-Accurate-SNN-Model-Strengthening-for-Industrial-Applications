import argparse

parser = argparse.ArgumentParser(description='stbp_based model forget')

parser.add_argument('--retraining',type=bool,default=False)
parser.add_argument('--forget-class', type=int, default=2, metavar='N',
                    help='forget-class')
parser.add_argument('--dataset-type',default='NEU_CLS_64',choices=['NEU_CLS_64','NEU_CLS','elpv'])
#######CNN-config#######
parser.add_argument('--time-steps-CNN',type=int,default=40)#40
########################
parser.add_argument('--simulation-time',type=int,default=120)#120
#######resnet-config#####
parser.add_argument('--time-steps-resnet',type=int,default=4)
########################
parser.add_argument('--inputsize',type=int,default=64)

parser.add_argument('--regenerate-NEU-64-or-not',default='N',choices=['Y','N'],help='regenerate only when changing the train_test_split size')
parser.add_argument('--regenerate-elpv-or-not',default='N',choices=['Y','N'],help='regenerate only when changing the train_test_split size')

parser.add_argument('--save-forget-model', action='store_true', default=True,
                    help='For Saving the current Forget Model')

parser.add_argument('--batch-size', type=int, default=32, metavar='N',#16 for NEU-CLS-200
                    help='input batch size for training (default: 32)')
parser.add_argument('--forget-batch', type=int, default=32, metavar='N',
                    help='batch size for model forget (default: 32)')
parser.add_argument('--spikes-track',type=bool,default=False)
parser.add_argument('--numclass', type=int, default=6, metavar='N',
                    help='numclass')
parser.add_argument('--first-train',default=True,choices=[True,False])
parser.add_argument('--NEU-CLS-traindataset-size', type=int, default=1440, metavar='N',#25 for model forget/10 for dvs-cifar10 train   #100 for shallow forget
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',#100
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--target-train-epochs', type=int, default=70, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--model-forget-epochs', type=int, default=3, metavar='N',
                    help='number of epochs to forget (default: 20)')
parser.add_argument('--break-point', type=int, default=10, metavar='N',#set the break point for saving model
                    help='number of epochs to train (default: 10)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer_select',default='Adam',choices=['Adam','SGD'])

parser.add_argument('--cuda', action='store_true', default=True,
                    help='CUDA training')
parser.add_argument('--save-pretrained-model',type=bool,default=True)
parser.add_argument('--log-interval', type=int, default=1,metavar='N',#1 9
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr-CNN',type=float,default=1e-4)#initial learning rate
parser.add_argument('--lr-forget',type=float,default=1e-3)#initial forget learning rate 5e-4 for NEU-CLS-64

parser.add_argument('--lambd',type=float,default=0.01)#KL-divergence term factor

parser.add_argument('--subset-or-class',default=False,choices=[True,False],help='subset forgotten (True) or class forgotten (False)')
###########random seed setting#############
parser.add_argument('--random-seed-for-NEU64-traintestsplit',type=int,default=613)
parser.add_argument('--random-seed-for-elpv-traintestsplit',type=int,default=521)
parser.add_argument('--random-seed-for-forgetdata-split',type=int,default=442)
############################################

args = parser.parse_args()

