


########## ---------- Import libaries ---------- ##########
import os
import numpy as np
import pandas as pd
import datetime

import torch
import torch.nn as nn
torch.cuda.empty_cache()

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utilities import Cardiac_Data_LA_2D
from utilities import distort_minibatch, separate_target_from_input
from utilities import dice_loss, dice_LV, dice_RV, dice_LV_Myo
from utilities import visualize_LA_slice_and_segment
from torchsummary import summary

import torchsummary

########## ---------- Variables ---------- ##########

from Pure_SA_3D_architecture import SA_UNet as Model

'''
model = Model()
Net_summary = summary(model, input_size = [(1, 16, 256, 256)], batch_size = -1)

f = open("SA_Net_summary_2Convlayers.txt", "w")
f.write(str(Net_summary))
f.close()
'''


from Pure_2D_architecture import UNet_2D as Model

model = Model()
Net_summary = summary(model, input_size = [(1, 1, 256, 256)], batch_size = -1)

f = open("LA_Net_summary_2Convlayers.txt", "w")
f.write(str(Net_summary))
f.close()


'''
from xUNet_architecture import xUNet as Model

model = Model()
Net_summary = summary(model, input_size = [(1, 1, 16, 256, 256), (1, 1, 256, 256)], batch_size = -1)

f = open("xNet_light_summary.txt", "w")
f.write(str(Net_summary))
f.close()

'''
'''
from P_HNN_SALA_aimal_1 import P_HNN_SALA_aimal_1 as Model

model = Model()
Net_summary = summary(model, input_size = [(1, 1, 16, 256, 256), (1, 1, 256, 256)], batch_size = -1)

f = open("PHNN_3D_SALA_summary.txt", "w")
f.write(str(Net_summary))
f.close()
'''

'''

from FusionBox import FusionBox_Net as Model

model = Model()
Net_summary = summary(model, input_size = [(1, 256, 4, 16, 16), (1, 1280, 16, 16)])

f = open("FusionBox_summary.txt", "w")
f.write(str(Net_summary))
f.close()
'''
'''
from P_HNN_SA_aimal import P_HNN_aimal_SA_2 as Model
model = Model()
Net_summary = summary(model, input_size = [(1, 256, 4, 16, 16), (1, 1280, 16, 16)])

f = open("P_HNN_SA_3D_2Convlayers_summary.txt", "w")
f.write(str(Net_summary))
f.close()
'''