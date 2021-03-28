from tqdm import tqdm
import numpy as np
import os, argparse
from utils import settings_parser


##### SETTINGS #####
setup_file = settings_parser.get_setup_file()

settings_dataset = settings_parser.get_settings('Dataset')
settings_model = settings_parser.get_settings('Model')

nVal = int(settings_model['nval'])
root_dir = settings_dataset['dataset_path']
dataset = settings_dataset['dataset_type']
name = settings_dataset['dataset_type']



data = os.path.join(root_dir, dataset, 'preprocessed',name)
train = np.load(data+'/train.npy')

if not os.path.exists(os.path.join(data,'points_train')):
    os.makedirs(os.path.join(data,'points_train'))

if not os.path.exists(os.path.join(data,'points_val')):
    os.makedirs(os.path.join(data,'points_val'))

if not os.path.exists(os.path.join(data,'points_test')):
    os.makedirs(os.path.join(data,'points_test'))


for i in tqdm(range(len(train)-nVal)):
    np.save(os.path.join(data,'points_train','{0}.npy'.format(i)),train[i])
for i in range(len(train)-nVal,len(train)):
    np.save(os.path.join(data,'points_val','{0}.npy'.format(i)),train[i])
    
test = np.load(data+'/test.npy')
for i in range(len(test)):
    np.save(os.path.join(data,'points_test','{0}.npy'.format(i)),test[i])
    
files = []
for r, d, f in os.walk(os.path.join(data,'points_train')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_train.npy'),files)

files = []
for r, d, f in os.walk(os.path.join(data,'points_val')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_val.npy'),files)

files = []
for r, d, f in os.walk(os.path.join(data,'points_test')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_test.npy'),files)

