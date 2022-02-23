import torch
import os
import copy
from collections import *

state_dict = 'D:\Gaze_estimator_implementation\experiment\eth_256_feature_extractor\gaze_model_best.pt'

state_dict = torch.load(state_dict, map_location=lambda storage, loc: storage)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if k[:6] == 'model.':
        name = k[6:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
    
torch.save(new_state_dict, './key_changed_gaze.pt')