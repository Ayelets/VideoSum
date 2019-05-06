import numpy as np
from scipy.io import loadmat
import os
import json
import h5py
import pickle
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import inception_v3
#from vids2features import *


'''
this code takes a dataset of videos saved as an h5 file, features saved with np.savetxt with 'delimiter=','
and replaces the 'features' index for each video 
'''

def load_features(txt_path):
    '''
    loads 'features' txt file (assumes that file was saved by np.savetext with delimiter=',')
    :param txt_path: path to txt file
    :return: features (ndarray with shape n_features x 2048)
    '''
    features = np.loadtxt(txt_path, delimiter=',')
    return features


def load_dataset(path):
    '''
    loads dataset (assumes that the dataset is an h5 file)
    :param path: path to the dataset
    :return: h5 file object with read/write privileges
    '''
    print('opening dataset: {}'.format(path))
    d_set = h5py.File(path, 'r')
    return d_set


def update_dataset(h5_path, features_dir, result_path, map):

    h5 = load_dataset(h5_path)
    hf = h5py.File(result_path, 'w')

    feature_lst = sorted(os.listdir(features_dir))
    vid_lst = sorted(list(h5.keys()))


    print('prepering to update dataset {}'.format(h5_path))
    h5_keys = sorted(list(map.keys()))
    print('h5 keys:\n{}'.format(h5_keys))
    print(feature_lst)
    print(vid_lst)
    attr_lst = sorted(list(h5[vid_lst[0]].keys()))
    print('attr list: \n{}'.format(attr_lst))
    # groups_lst = ['group{}'.format(k) for k in range(len(vid_lst))]
    # print('groups list: \n{}'.format(groups_lst))

    for vid in vid_lst:
        print('copying h5[{}]'.format(vid))
        print('getting {}'.format(features_dir + map[vid]))
        features = load_features(features_dir + map[vid][:-4] + '.csv')
        g = hf.create_group(vid)
        for j in range(len(attr_lst)):
            if attr_lst[j] != 'features':
                g.create_dataset(attr_lst[j], data=h5[vid][attr_lst[j]])
            else:
                print('adding new feature: old features shape: {}, new features shape: {}'.format(h5[vid]['features'][()].shape, features.shape))
                g.create_dataset(attr_lst[j], data=features)
    return hf


def get_vids_nframes(vids_folder, vids_lst):
    d = {}
    for vid in vids_lst:
        vid_path = vids_folder + vid
        cap = cv2.VideoCapture(vid_path)
        if cap is None:
            print('problem in get_vids_nframes video {}:\n fool, cap is None!!!'.format(vid))
        else:
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            d[nframes] = vid
        cap.release()
    return d


def get_h5_nframes(h5):
    vids_lst = sorted(list(h5.keys()))
    d = {}
    for v in vids_lst:
        nframes = int(h5[v]['n_frames'][()])
        d[nframes] = v
    return d


def map_vids_with_h5_keys(vids_nframes, h5_nframes):
    mapping = {}
    nframes = sorted(vids_nframes.keys())
    nframes2 = sorted(h5_nframes.keys())
    map = {}
    for i in range(len(nframes)):
        n1 = nframes[i]
        n2 = nframes2[i]
        if abs(n1 - n2) > 1:
            print('problem in map_vids_with_h5_keys:')
            print('fuck it all!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            break
        else:
            map[h5_nframes[n2]] = vids_nframes[n1]
    return map

def vids2features(result_dir, vids_path, model, ds = 15):

    f = os.listdir(vids_path)
    f.sort()
    vids_lst = [vids_path+"/" + s for s in f if s[-1] == "4"]

    for vid_path in vids_lst:
        cap = cv2.VideoCapture(vid_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames=[]
        corrupt_frames = []

        for i in range(n_frames):
            ret, frame = cap.read()
            if frame is not None:
                if i % ds == 0:
                    frame = cv2.resize(frame, (299, 299))
                    frames.append(frame)
            else:
                corrupt_frames.append(i)
                frames.append(frames[-1])

        if len(corrupt_frames) > 0:
            print('fool, farmes {} in vide {} are corrupted'.format(i, vids_path))

        predictions = model.predict(np.array(frames))
        n , m = predictions.shape
        print(predictions.shape)
        vid_name=vid_path.split("/")[-1]
        np.savetxt(result_dir + vid_name[:-4]+".csv", predictions, delimiter=",")
        print('completed file {} (num of frames {}), num of features {}'.format(vid_name[:-4]+".csv", n, m))


def create_dataset(dataset_folder_path, dataset_name, features_path, results_path,model):

    dataset_folder_path = 'datasets_original/'
    dataset_name = 'eccv16_dataset_summe_google_pool5.h5'

    features_path = 'features/'
    results_path = 'results/fixed_datasets/'
 #--------------------------------------------------
    # fix sumMe h5:
 #-------------------------------------------------

    print('preparing to fix sumMe features:')
    print('--------------------------------------')
    dataset_path = dataset_folder_path + dataset_name
    result_sumMe_path = results_path + 'dataset_summe_normed_inception_v3.h5'
    vids_dir = 'videos/'


    vids_lst = sorted([vid for vid in os.listdir(vids_dir) if vid[-4:] == '.mp4'])
    print(vids_dir)
    print(vids_lst)
    print()

    groups_lst = ['video_' + str(i) for i in range(1, len(vids_lst) + 1)]
    print(groups_lst)
    map = {}
    for i in range(len(groups_lst)):
        map[groups_lst[i]] = vids_lst[i]

    print(map)

    vids2features(features_path, "videos", model)

    hf = update_dataset(dataset_path, features_path, result_sumMe_path, map=map)
    h5 = load_dataset(dataset_path)
    hf.close()
    h5.close()



