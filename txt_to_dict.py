import os
import pickle
from tqdm import tqdm

with open("./dataset/eth_label.txt") as f:
    lines = f.readlines()
    

label_dict = {}

for line in tqdm(lines):
    name = line.split(' ')[0]
    gaze_label = line.split(' ')[2].split(',')
    gaze_label[-1] = gaze_label[-1][:-2]
    for i, l in enumerate(gaze_label):
        gaze_label[i] = float(l)
    
    label_dict[name] = gaze_label
    

with open("./dataset/eth_label_dict.pickle", "wb") as f:
    pickle.dump(label_dict, f)


with open("./dataset/eth_label_dict.pickle", "rb") as f:
    label_dict = pickle.load(f)

print(label_dict['0000_1'])
print(type(label_dict['0000_1']))
