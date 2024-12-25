import pickle, logging, numpy as np
from torch.utils.data import Dataset


class NTU_Feeder(Dataset):
    def __init__(self, phase, dataset_path, inputs, num_frame, connect_joint, debug, **kwargs):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint
        data_path = '{}/{}_data.npy'.format(dataset_path, phase)
        label_path = '{}/{}_label.pkl'.format(dataset_path, phase)
        try:
            self.data = np.load(data_path, mmap_mode='r') # N,C, T, V, M
            with open(label_path, 'rb') as f:
                self.name, self.label, self.seq_len = pickle.load(f, encoding='latin1')
        except:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.name = self.name[:300]
            self.seq_len = self.seq_len[:300]

        #只输出指定标签的内容
        # !!!!验证结果时输出分布，训练时注释掉
        # validIndex = np.where(np.array(self.label)==0)[0].tolist()
        # self.data=self.data[validIndex]
        # labelTemp=[]
        # nameTemp=[]
        # seq_lenTmp=[]
        # for i in validIndex:
        #     labelTemp.append(self.label[i])
        #     nameTemp.append(self.name[i])
        #     seq_lenTmp.append(self.seq_len[i])
        # self.label=labelTemp
        # self.name=nameTemp
        # self.seq_len=seq_lenTmp

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx]) # 这是获取某一个目标，然后对此目标预处理 C、T、V、M
        label = self.label[idx]
        name = self.name[idx]
        # seq_len = self.seq_len[idx]
        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        joint, velocity, bone = self.multi_input(data[:,:self.T,:,:])
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        data_new = np.stack(data_new, axis=0)
        return data_new, label, name


    def multi_input(self, data):

        """
        num_node = 10
            neighbor_link = [(0, 1), (1, 5), (2, 5), (3, 2),
                             (4, 3), (6, 5), (7, 5), (8, 7), (9, 8)]

            #为了构造输入的第三种特征，需要定义每根骨骼（以节点标号表征）相对于哪个节点计算
            connect_joint = np.array([1, 5, 5, 2, 3, 5, 5, 5, 7, 8])
            parts = [
                np.array([7, 8, 9]),  # left_arm
                np.array([2, 3, 4]),  # right_arm
                np.array([0, 1, 5, 6])  # torso
            ]
        """

        C, T, V, M = data.shape # M: 1 T: 45 V: 10 C: 2
        joint = np.zeros((C*2, T, V, M))
        velocity = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        joint[:C,:,:,:] = data
        for i in range(V): # 关键点个数，归一化后关键点到中心关节点的距离
            #joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, 5, :]
        for i in range(T-2):
            velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:] # slow motion
            velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:] # fast motion
        for i in range(len(self.conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length) # 前面C个保存的是骨骼间长度，后面C个保存的是角度
        return joint, velocity, bone


class NTU_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 2, self.T, self.V, self.M))
        for i, name in enumerate(names):
            with open(name, 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        fr.readline()
                        joint_num = int(fr.readline())
                        for joint in range(joint_num):
                            v = fr.readline().split(' ')
                            if joint < self.V and person < self.M:
                                location[i,0,frame,joint,person] = float(v[5])
                                location[i,1,frame,joint,person] = float(v[6])
        return location
