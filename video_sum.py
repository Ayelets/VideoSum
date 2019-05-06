import cv2
import pickle
from keras.applications.inception_v3 import InceptionV3
import numpy as np
import os
from cpd_auto import cpd_auto

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2





model = InceptionV3(include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling='avg',
                    classes=1000)


f=os.listdir("videos")
f.sort()
f=["videos/"+ s for s in f if s[-1]=="4"]

for vidio_path in f:
    cap = cv2.VideoCapture(vidio_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame=None
    frames=[]


    for index in range(length):
        ret, frame = cap.read()

        if index%15==0:
            try:
                frame = cv2.resize(frame, (299, 299))
                frames.append(frame)
            except:
                print(index)
                print(vidio_path)
                frames.append(frames[-1])
    print(vidio_path[:-4]+".csv")
    predictions = model.predict(np.array(frames))
    print(predictions.shape)
    np.savetxt(vidio_path[:-4]+".csv", predictions, delimiter=",")
    #
    # X = predictions
    # n = X.shape[0]
    # K = np.dot(X, X.T)
    # cps, scores = cpd_auto(K, length//(fps*5), 1)
    #
    # j=0
    # cap = cv2.VideoCapture(vidio_path)
    # cps = np.concatenate(([0], cps))
    # cps = np.concatenate((cps, [n]))
    # change_points = np.array([[cps[i], cps[i + 1] - 1] for i in range(cps.shape[0] - 1)], dtype=np.int64)
    # n_frames = np.array(n, dtype=np.int64)
    # #np.savetxt(vidio_path[:-4] + ".change_points", change_points, delimiter=",")
    # #b=np.loadtxt(vidio_path[:-4] + ".change_points", delimiter=',')
    #
    #
    # #print(predictions.shape)
    # cap.release()

'''    for i in range(length):
        if i>cps[j]:
            j+=1
        ret, frame = cap.read()
        cv2.putText(frame, str(j),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('frame', frame)
        cv2.waitKey(10)'''


#vid2names=pickle.load( open("vid2names.pickle","rb"))


#for v,n in vid2names:


#    if frame_list[index]>0.5:
#       sum_vid.append(frame)
#           print("a")


    # height, width, channels = frame.shape
    #
    # out = cv2.VideoWriter("sum_"+move_name[:-4]+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (height, width))
    #
    # for i in range(len(sum_vid)):
    #     out.write(sum_vid[i])
    #     cv2.imshow('frame', sum_vid[i])
    #     cv2.waitKey(30)
    #
    # out.release()
    #
