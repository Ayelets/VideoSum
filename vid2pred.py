from AONet import*
from keras.applications.inception_v3 import InceptionV3
import cv2
from config import  *
from cpd_auto import cpd_auto
import matplotlib.pyplot as plt
from keras.applications import inception_v3

def get_user_sumery_list():
    key_list=["video_"+str(i) for i in range(1,26)]
    d_set = h5py.File("datasets/dataset_summe_inception_v3.h5", 'r')
    user_sumery_list=[]
    for key in key_list:
        user_sumery_list.append(np.array(d_set[key]['gtscore']))
    return user_sumery_list

def change_points(predictions,length,fps,vidio_path):
    X = predictions
    n = X.shape[0]
    K = np.dot(X, X.T)
    cps, scores = cpd_auto(K, length//(fps*5), 1)
    cap = cv2.VideoCapture(vidio_path)
    cps = np.concatenate(([0], cps))
    cps = np.concatenate((cps, [n]))
    change_points = np.array([[cps[i], cps[i + 1] - 1] for i in range(cps.shape[0] - 1)], dtype=np.int64)
    return change_points


def video_summarizer(original_videos_folder,trained_model="models_new/2_73.822.pth.tar"):
    hps =HParameters()
    ao = AONet(hps)
    ao.initialize()
    ao.load_model(trained_model)

    model = InceptionV3(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling='avg',
                        classes=1000)

    f=os.listdir(original_videos_folder)
    f.sort()
    f=["videos/"+ s for s in f if s[-1]=="4"]

    #user_sumery_list=get_user_sumery_list()
    #f=["20190425_165804.mp4"]

    for ind_vid,vidio_path in enumerate(f):
        cap = cv2.VideoCapture(vidio_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame=None
        frames=[]


        for index in range(length):
            ret, frame = cap.read()

            try:
                frame = cv2.resize(frame, (299, 299))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            except:
                print(index)
                print(vidio_path)
                frames.append(frames[-1])
        frames=np.array(frames)
        frames = inception_v3.preprocess_input(frames)
        predictions = model.predict(frames)

        pred_sub_sump=predictions[np.arange(0,len(predictions),15)]

        seq = pred_sub_sump
        seq = torch.from_numpy(seq).unsqueeze(0)

        seq = seq.float().cuda()

        #predict

        y, att_vec = ao.model(seq, seq.shape[1])

        cps = change_points(predictions,length,fps,vidio_path)
        num_frames = length
        positions = np.array(range(0,num_frames,15),dtype=np.int64)
        probs = y[0].detach().cpu().numpy()
        #user_sumery=user_sumery_list[ind_vid]
        # plt.title(vidio_path[7:-4])
        # plt.bar(list(range(len(probs))), probs/np.sum(probs),alpha=0.4,color="r", linewidth=0.5)
        # plt.bar(list(range(len(user_sumery))), user_sumery/np.sum(user_sumery),alpha=0.4,color="b", linewidth=0.5)
        # plt.savefig(vidio_path[7:-4]+"plt"+"png")
        # #plt.show()


        nfps=[j-i+1 for i,j in cps]

        machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        cap = cv2.VideoCapture(vidio_path)
        out = cv2.VideoWriter(vidio_path[:-4]+".avi", fourcc, fps, (int(cap.get(3)),int(cap.get(4))))
        for index in range(length):
             ret, frame = cap.read()
             if machine_summary[index]>0.5:
                 out.write(frame)



