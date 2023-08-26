from train_target_model_SCNN import train_SCNN
from model_forget import SPMU
from forget_model_test import model_test
from retraining import retrain
from utils import *

import os

traindataset=args.dataset_type
Net_structure='SCNN_'
target_model_path=Net_structure+traindataset+'_'+'target_model.pt'
forget_model_path=Net_structure+traindataset+'_'+'forget_model.pt'
unlearn_command=True
target_training_command=False
retrain_command=False
#train model#################################################
if target_model_path not in os.listdir('snn_forget_industry/experiment_log/target_model/') or target_training_command: 
    train_SCNN(traindataset)
                        #some commands    ##CUDA_VISIBLE_DEVICES=1 python snn_forget_industry/main.py --dataset-type 'NEU_CLS_64' --inputsize 64 --numclass 9 --lambd 0.001 --forget-class 1 --lr-forget 5e-4
                                            ##CUDA_VISIBLE_DEVICES=1 python snn_forget_industry/main.py --dataset-type 'NEU_CLS' --inputsize 128 --numclass 6 --lambd 0.001 --forget-class 2 --lr-forget 5e-4
                                            
##############################################################

#unlearn model################################################
if forget_model_path not in os.listdir('snn_forget_industry/experiment_log/unlearned_model/') or unlearn_command:#directly implement the model forget procedure if target model has already been trained
    print('target model has been trained')

    SPMU('snn_forget_industry/experiment_log/target_model/'+target_model_path,'snn_forget_industry/experiment_log/unlearned_model/'+forget_model_path)
##############################################################

#retrain model################################################
if forget_model_path not in os.listdir('snn_forget_industry/experiment_log/retrained_model/') or retrain_command:
    print('target model has been trained')
    args.retraining=True
    retrain('snn_forget_industry/experiment_log/retrained_model/'+forget_model_path)
    args.retraining=False
##############################################################


model_path='snn_forget_industry/experiment_log/unlearned_model/'+forget_model_path
model_path2='snn_forget_industry/experiment_log/retrained_model/'+forget_model_path
#test model###################################################
print('forget_model loaded')













#model_test('snn_forget_industry/experiment_log/target_model/'+target_model_path)#before unlearning
model_test(model_path)#after unlearning through KL
#model_test(model_path2)#after unlearning through retraining
##############################################################
