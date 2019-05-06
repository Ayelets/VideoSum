import cv2
from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
import numpy as np
import os


#--------------------------------------------------------------
# input video path
sumMe_vids_path = 'videos/'
tvSum_vids_path = '/home/itziknanikashvili/Desktop/video_summ/tvSum/ydata-tvsum50-v1_1/video/'
features_save_path = '/home/itziknanikashvili/PycharmProjects/video_summ/VASNet/results/features/sumMe/'
#--------------------------------------------------------------

model = InceptionV3(include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling='avg',
                    classes=1000)

vids_lst = os.listdir(sumMe_vids_path)
print(vids_lst)
vids_lst = sorted([vid for vid in vids_lst if vid[-1] == "4"])
print(vids_lst)
for vid in vids_lst:
    vid_path = sumMe_vids_path + vid
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('proceesing {}: n_frames = {}'.format(vid, n_frames))

    if cap is None:
        print('couldnt capture {}'.format(vid))
    if frame is None:
        print('couldnt capture frame in {}'.format('0', vid))
    h, w, c = (299, 299, 3)

    frames = np.zeros((1 + (n_frames-1)//15, h, w, c), dtype=np.uint8)

    # create video frames tensor np.array with shape (nframes, hight, width, channels) (also converts BGR2RGB)
    for i in range(n_frames):
        if i == n_frames-1 and frame is not None:
            print('ok vid {}'.format(vid))
            break

        if frame is not None:
            if i % 15 == 0:
                frame = cv2.resize(frame, (299, 299))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[i//15, :, :, :] = frame
        else:
            print('frame {} is empty'.format(i), end=', ')
            if i % 15 ==0 :
                frames[i//15, :, :, :] = frames[i//15 - 1, :, :, :]

        ret, frame = cap.read()

    print('\n')

    # normalize frames tensor for imagenet
    frames = inception_v3.preprocess_input(frames)
    predictions = model.predict(frames)
    save_path = features_save_path + vid[:-4] + '.txt'
    print('saving {}'.format(save_path))
    np.savetxt(save_path, predictions, delimiter=',')

    #print(predictions.shape)
    cap.release()

