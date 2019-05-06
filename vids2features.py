
import cv2
import numpy as np
import os


def vids2features(result_dir, vids_path, model, ds = 15):

    vids_lst = os.listdir(vids_path)

    for vid in vids_lst:
        vid_path = vids_path + vid
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
            print('fool, farmes {} in vide {} are corrupted'.format(i, vid))

        predictions = model.predict(np.array(frames))
        n , m = predictions.shape
        print(predictions.shape)
        np.savetxt(result_dir + vid[:-4]+".csv", predictions, delimiter=",")
        print('completed file {} (num of frames {}), num of features {}'.format(vid[:-4]+".csv", n, m))
