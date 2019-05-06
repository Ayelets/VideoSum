__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"


import torch
from torchvision import transforms
import numpy as np
import time
import glob
import random
import argparse
import h5py
import json
import torch.nn.init as init
import pickle

from config import  *
from sys_utils import *
from vsum_tools import  *
from vasnet_model import  *
from AONet import*
from replace_features import*
from vid2pred import*

def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)



def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    dataset_type_str = '' if dataset_type == '' else dataset_type + '_'
    weights_filename = path + '/models/{}_{}splits_{}_*.tar.pth'.format(dataset_name, dataset_type_str, split_id)
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ''

    # Get the first weights file in the dir
    weights_filename = weights_filename[0]
    splits_file = path + '/splits/{}_{}splits.json'.format(dataset_name, dataset_type_str)

    return weights_filename, splits_file




#==============================================================================================

def train(hps):
    dataset_folder_path = 'datasets_original/'
    sumMe_dataset_name = 'eccv16_dataset_summe_google_pool5.h5'

    features_path = 'features/sumMe/'
    results_path = 'results/fixed_datasets/'
    feature_model = InceptionV3(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling='avg',
                        classes=1000)

    #Create a data set with an feature extranet from given network (feature_model)
    #replace the given feature-extraction in h5 file
    create_dataset(dataset_folder_path, sumMe_dataset_name, features_path, results_path,feature_model)
    os.makedirs(hps.output_dir, exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'code'), exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'models'), exist_ok=True)
    os.system('cp -f splits/*.json  ' + hps.output_dir + '/splits/')
    os.system('cp *.py ' + hps.output_dir + '/code/')

    # Create a file to collect results from all splits
    f = open(hps.output_dir + '/results.txt', 'wt')

    ao = AONet(hps)
    ao.initialize()
    ao.load_model("models_new/2_73.822.pth.tar")

    for split_filename in hps.splits:
        dataset_name, dataset_type, splits = parse_splits_filename(split_filename)

        # For no augmentation use only a dataset corresponding to the split file
        datasets = None
        if dataset_type == '':
            datasets = hps.get_dataset_by_name(dataset_name)

        if datasets is None:
            datasets = hps.datasets

        f_avg = 0
        n_folds = len(splits)
        ao.load_datasets(datasets=datasets)
        ao.load_split_file(splits_file=split_filename)
        for split_id in range(n_folds):

            ao.select_split(split_id=split_id)

            fscore, fscore_epoch = ao.train(output_dir=hps.output_dir)
            f_avg += fscore

            # Log F-score for this split_id
            f.write(split_filename + ', ' + str(split_id) + ', ' + str(fscore) + ', ' + str(fscore_epoch) + '\n')
            f.flush()

            # Save model with the highest F score
            _, log_file = os.path.split(split_filename)
            log_dir, _ = os.path.splitext(log_file)
            log_dir += '_' + str(split_id)
            log_file = os.path.join(hps.output_dir, 'models', log_dir) + '_' + str(fscore) + '.tar.pth'

            os.makedirs(os.path.join(hps.output_dir, 'models', ), exist_ok=True)
            os.system('mv ' + hps.output_dir + '/models_temp/' + log_dir + '/' + str(fscore_epoch) + '_*.pth.tar ' + log_file)
            os.system('rm -rf ' + hps.output_dir + '/models_temp/' + log_dir)

            print("Split: {0:}   Best F-score: {1:0.5f}   Model: {2:}".format(split_filename, fscore, log_file))

        # Write average F-score for all splits to the results.txt file
        f_avg /= n_folds
        f.write(split_filename + ', ' + str('avg') + ', ' + str(f_avg) + '\n')
        f.flush()

    f.close()


if __name__ == "__main__":
    print_pkg_versions()

    parser = argparse.ArgumentParser("PyTorch implementation of paper \"Summarizing Videos with Attention\"")
    parser.add_argument('-r', '--root', type=str, default='', help="Project root directory")
    parser.add_argument('-d', '--datasets', type=str, help="Path to a comma separated list of h5 datasets")
    parser.add_argument('-s', '--splits', type=str, help="Comma separated list of split files.")
    parser.add_argument('-t', '--train', action='store_true', help="Train")
    parser.add_argument('-v', '--verbose', action='store_true', help="Prints out more messages")
    parser.add_argument('-o', '--output-dir', type=str, default='data', help="Experiment name")
    args = parser.parse_args()

    # MAIN
    #======================
    hps = HParameters()
    hps.load_from_args(args.__dict__)

    print("Parameters:")
    print("----------------------------------------------------------------------")
    print(hps)

    #creat_data_for_train
    #creat featcher vector(inseption3 imagenet) for movie and replace it whith the corent featcher vector
    path_h5_files="datasets_original/eccv16_dataset_summe_google_pool5.h5"
    path_vidios_files="videos_summe"


    if hps.train:
        train(hps)
    else:
        #insert path to videos folder the summary save as avi file
        video_summarizer("videos", trained_model="models_new/2_73.822.pth.tar")

