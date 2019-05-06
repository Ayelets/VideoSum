import random
import argparse
import h5py
import json
import torch.nn.init as init
import pickle
from sys_utils import *
from vsum_tools import  *
from vasnet_model import  *
import glob


def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == 'splits':
        # Split type is not present
        dataset_type = ''

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits

class AONet:

    def __init__(self, hps: HParameters):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose


    def fix_keys(self, keys, dataset_name = None):
        """
        :param keys:
        :return:
        """
        # dataset_name = None
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))

        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(len(self.datasets))

                key_name = dataset_name+'/'+key
                keys_out.append(key_name)
            else:
                keys_out.append(key)

        return keys_out


    def load_datasets(self, datasets = None):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """
        if datasets is None:
            datasets = self.hps.datasets

        datasets_dict = {}

        for dataset in datasets:
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            print("Loading:", dataset)
            # dataset_name = base_filename.split('_')[2]
            # print("\tDataset name:", dataset_name)
            datasets_dict[base_filename] = h5py.File(dataset, 'r')

        self.datasets = datasets_dict
        return datasets_dict


    def load_split_file(self, splits_file):

        self.dataset_name, self.dataset_type, self.splits = parse_splits_filename(splits_file)
        n_folds = len(self.splits)
        self.split_file = splits_file
        print("Loading splits from: ",splits_file)

        return n_folds


    def select_split(self, split_id):
        print("Selecting split: ",split_id)

        self.split_id = split_id
        n_folds = len(self.splits)
        assert self.split_id < n_folds, "split_id (got {}) exceeds {}".format(self.split_id, n_folds)

        split = self.splits[self.split_id]
        self.train_keys = split['train_keys']
        self.test_keys = split['test_keys']

        dataset_filename = self.hps.get_dataset_by_name(self.dataset_name)[0]
        _,dataset_filename = os.path.split(dataset_filename)
        dataset_filename,_ = os.path.splitext(dataset_filename)
        self.train_keys = self.fix_keys(self.train_keys, dataset_filename)
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return



    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        return


    def initialize(self, cuda_device=None):
        rnd_seed = 12345
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

        self.model = VASNet()
        self.model.eval()
        #self.model.apply(weights_init)
        #print(self.model)

        cuda_device = cuda_device or self.hps.cuda_device

        if self.hps.use_cuda:
            print("Setting CUDA device: ",cuda_device)
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(rnd_seed)

        if self.hps.use_cuda:
            self.model.cuda()

        return


    def get_data(self, key):
        key_parts = key.split('/')
        assert len(key_parts) == 2, "ERROR. Wrong key name: "+key
        dataset, key = key_parts
        return self.datasets[dataset][key]

    def lookup_weights_file(self, data_path):
        dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
        weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''

        # Get the first weights filename in the dir
        weights_filename = weights_filename[0]
        splits_file = data_path + '/splits/{}_{}splits.json'.format(self.dataset_name, dataset_type_str)

        return weights_filename, splits_file


    def train(self, output_dir='EX-0'):

        print("Initializing VASNet model and optimizer...")
        self.model.train()

        criterion = nn.MSELoss()
        #criterion =nn.CrossEntropyLoss()
        #criterion = nn.L1Loss()

        if self.hps.use_cuda:
            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.hps.lr[0], weight_decay=self.hps.l2_req)

        print("Starting training...")

        max_val_fscore = 0
        max_val_fscore_epoch = 0
        train_keys = self.train_keys[:]

        lr = self.hps.lr[0]
        for epoch in range(self.hps.epochs_max):

            print("Epoch: {0:6}".format(str(epoch)+"/"+str(self.hps.epochs_max)), end='')
            self.model.train()
            avg_loss = []

            random.shuffle(train_keys)

            for i, key in enumerate(train_keys):
                dataset = self.get_data(key)
                seq = dataset['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                target = dataset['gtscore'][...]
                target = torch.from_numpy(target).unsqueeze(0)

                # Normalize frame scores
                target -= target.min()
                target /= target.max()

                if self.hps.use_cuda:
                    seq, target = seq.float().cuda(), target.float().cuda()

                seq_len = seq.shape[1]
                y, _ = self.model(seq,seq_len)
                loss_att = 0

                loss = criterion(y, target)
                #loss2 = y.sum()/seq_len
                loss = loss + loss_att
                self.optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.1)
                self.optimizer.step()
                #avg_loss.append([float(loss), float(loss_att)])
                avg_loss.append([float(loss)])

            # Evaluate test dataset
            val_fscore, video_scores = self.eval(self.test_keys)
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                max_val_fscore_epoch = epoch
                print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                # Save model weights
                path, filename = os.path.split(self.split_file)
                base_filename, _ = os.path.splitext(filename)
                path = os.path.join(output_dir, 'models_temp')
                os.makedirs("models_new", exist_ok=True)
                filename = str(epoch)+'_'+str(round(val_fscore*100,3))+'.pth.tar'
                torch.save(self.model.state_dict(), os.path.join("models_new", filename))

            avg_loss = np.array(avg_loss)
            print("   Train loss: {0:.05f}".format(np.mean(avg_loss[:, 0])), end='')
            print('   Test F-score avg/max: {0:0.5}/{1:0.5}'.format(val_fscore, max_val_fscore))

            if self.verbose:
                video_scores = [["No", "Video", "F-score"]] + video_scores
                print_table(video_scores, cell_width=[3,40,8])




        return max_val_fscore, max_val_fscore_epoch


    def eval(self, keys, results_filename=None):

        self.model.eval()
        summary = {}
        att_vecs = {}
        with torch.no_grad():
            for i, key in enumerate(keys):
                data = self.get_data(key)
                # seq = self.dataset[key]['features'][...]
                seq = data['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)

                if self.hps.use_cuda:
                    seq = seq.float().cuda()

                #predict
                y, att_vec = self.model(seq, seq.shape[1])
                summary[key] = y[0].detach().cpu().numpy()
                att_vecs[key] = att_vec.detach().cpu().numpy()

        f_score, video_scores = self.eval_summary(summary, keys, metric=self.dataset_name,
                    results_filename=results_filename, att_vecs=att_vecs)

        return f_score, video_scores


    def eval_summary(self, machine_summary_activations, test_keys, results_filename=None, metric='tvsum', att_vecs=None):

        eval_metric = 'avg' if metric == 'tvsum' else 'max'

        if results_filename is not None:
            h5_res = h5py.File(results_filename, 'w')

        fms = []
        video_scores = []
        for key_idx, key in enumerate(test_keys):
            d = self.get_data(key)
            probs = machine_summary_activations[key]

            if 'change_points' not in d:
                print("ERROR: No change points in dataset/video ",key)

            cps = d['change_points'][...]
            num_frames = d['n_frames'][()]
            nfps = d['n_frame_per_seg'][...].tolist()
            positions = d['picks'][...]
            user_summary = d['user_summary'][...]

            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
            name=key.replace("/", "_")
            with open(name+"_machine_summary.pkl", 'wb') as f:
                pickle.dump(machine_summary, f)
            with open(name+"_user_summary.pkl", 'wb') as f:
                pickle.dump(user_summary, f)

            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            # Reporting & logging
            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])

            if results_filename:
                gt = d['gtscore'][...]
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=gt)
                h5_res.create_dataset(key + '/fm', data=fm)
                h5_res.create_dataset(key + '/picks', data=positions)

                video_name = key.split('/')[1]
                if 'video_name' in d:
                    video_name = d['video_name'][...]
                h5_res.create_dataset(key + '/video_name', data=video_name)

                if att_vecs is not None:
                    h5_res.create_dataset(key + '/att', data=att_vecs[key])

        mean_fm = np.mean(fms)

        # Reporting & logging
        if results_filename is not None:
            h5_res.close()

        return mean_fm, video_scores
