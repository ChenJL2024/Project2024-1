import os, pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .transformer import pre_normalization
import cv2
import glob
import random

class NTU_Reader():
    def __init__(self, args, root_folder, transform, ntu60_path, ntu120_path, **kwargs):
        self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 10
        self.max_person = 1
        self.select_person_num = 1
        self.dataset = args.dataset
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        # Set paths
        ntu_ignored = '{}/ignore.txt'.format(os.path.dirname(os.path.realpath(__file__)))
        if self.transform:
            self.out_path = '{}/transformed/{}'.format(root_folder, self.dataset)
        else:
            self.out_path = '{}/original/{}'.format(root_folder, self.dataset)
        U.create_folder(self.out_path)

        # Divide train and eval samples
        training_samples = dict()
        training_samples['ntu-xsub'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
        ]
        training_samples['ntu-xview'] = [2, 3]
        training_samples['ntu-xsub120'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        training_samples['ntu-xset120'] = set(range(2, 33, 2))
        self.training_sample = training_samples[self.dataset]

        # Get ignore samples
        try:
            with open(ntu_ignored, 'r') as f:
                self.ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
        except:
            logging.info('')
            logging.error('Error: Wrong in loading ignored sample file {}'.format(ntu_ignored))
            raise ValueError()


    def read_file(self, file):
        '''
        skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        with open(file_path, 'r') as fr:
            frame_num = int(fr.readline())
            for frame in range(frame_num):
                person_num = int(fr.readline())
                for person in range(person_num):
                    person_info = fr.readline().strip().split()
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        joint_info = fr.readline().strip().split()
                        skeleton[person,frame,joint,:] = np.array(joint_info[:self.max_channel], dtype=np.float32)
        '''
        skeleton=np.load(file) #N,C,T,V

        #return skeleton[:,:frame_num,:,:], frame_num
        if skeleton.shape[0] == 45:
            skeleton = skeleton.transpose(1,3,0,2)
        return skeleton, skeleton.shape[0]


    def gendata(self):
        sample_data = []
        sample_label = []
        sample_path = []
        sample_length = []
        numTotalSets=0
        # for file_path in glob.glob(pathname='./datasets/test_back_0/*.npy'):
        for file_path in glob.glob(pathname='./datasets/sit-stand/original/*.npy'):
            skeleton, lenLable = self.read_file(file_path) # skeleton shape: N,C,T,V
            # print(skeleton.shape)
            #if 'eep' in file_path:
            # if file_path[-5:] in ['0.npy','1.npy','2.npy','3.npy','4.npy','5.npy','6.npy','7.npy','8.npy','9.npy']:
            #    lable=0
            # else:
            #    lable=1
            # passon_numple = ['03_1.npy','03_2.npy', '03_3.npy', '03_4.npy', '03_5.npy', '03_6.npy', '03_7.npy', '0315_1.npy', '0315_2.npy',\
            #                  '0315_3.npy','0315_4.npy','0315_5.npy','0315_6.npy','0315_7.npy','0315_8.npy','0315_9.npy']
            passon = ['val_passon_15s.npy']
            back_numple = ['back.npy','back_0.npy','back_1.npy']
            peep_numple = ['peep_0.npy','peep_1.npy', 'peep.npy']
            stand_up = ['stand.npy','sit.npy','stand2.npy','stand3.npy']
            raise_up = ['raise_2.7.npy','raise1.npy']
            raise_down = ['raise_down.npy']
            sit = ['sit0.npy','sit1.npy','sit2.npy','sit3.npy']
            # stand_sit = ['sit0.npy','sit1.npy','stand2.npy','stand3.npy']
            sit_stand = ['sit_original.npy']
            stand_sit = ['sit2.npy','sit3.npy','stand0.npy','stand1.npy']
            posfile = file_path.split('/')[-1]
            if posfile in sit_stand:
                # if file_path[-5:] in ['l.npy']:
                lable = 0
            else:
                lable = 1
                print(posfile)

            '''
            elif 'assOn' in file_path:
               #lable=1
               continue
            else:
               lable=1
            '''

            '''
            if file_path[-10:] == 'normal.npy':
                for i in random.sample(range(lenLable),int(lenLable*0.95)):
                    sample_data.append(skeleton[i][:, :, :, None])
                    sample_path.append(file_path)
                    sample_label.append(lable)  # to 0-indexed
                    sample_length.append(300)
                    numTotalSets += 1
                print(f'sample Normal Done.')
                continue
            '''

            for i in range(lenLable):
                sample_data.append(skeleton[i][:,:,:,None])
                sample_path.append(file_path)
                sample_label.append(lable)  # to 0-indexed
                sample_length.append(300)
                numTotalSets+=1

        indexTrain=set(random.sample(range(numTotalSets),int(numTotalSets*0.8)))
        indexEval=set([i for i in range(numTotalSets)]) - indexTrain
        indexTrain=list(indexTrain)
        indexTrain.sort()
        indexEval = list(indexEval)
        indexEval.sort()

        phase='train'
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump(([sample_path[i] for i in indexTrain],\
                         [sample_label[i] for i in indexTrain],\
                         [sample_length[i] for i in indexTrain]), f)

        sample_dataOut = np.array([sample_data[i] for i in indexTrain])
        np.save('{}/{}_data.npy'.format(self.out_path, phase), sample_dataOut) # N, C, T, V, M

        phase = 'eval'
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump(([sample_path[i] for i in indexEval], \
                         [sample_label[i] for i in indexEval], \
                         [sample_length[i] for i in indexEval]), f)
            
        sample_dataOut = np.array([sample_data[i] for i in indexEval])
        np.save('{}/{}_data.npy'.format(self.out_path, phase), sample_dataOut) # N, C, T, V, M

    def start(self):
        self.gendata()
