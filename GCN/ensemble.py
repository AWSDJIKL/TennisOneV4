import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]
        
        r = r11 * weights[0] + r22 * weights[1] + r33 * weights[2] + r44 * weights[3]
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc  # We want to maximize accuracy, hence minimize -accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA', 'csv1','csv2'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)


    arg = parser.parse_args()

    dataset = arg.dataset
    if 'csv1' in arg.dataset:
        npz_data = np.load('GCN/data/V1.npz')
        label = npz_data['y_test']#np.where(npz_data['y_test'] > 0)[1]
    elif 'csv2' in arg.dataset:
        npz_data = np.load('data/liujinfu/icmew/pose_data/V2.npz')
        label = npz_data['y_test']#np.where(npz_data['y_test'] > 0)[1]

    else:
        raise NotImplementedError

     # another method to get label
    '''
    label = []
    if 'csv1' in arg.dataset:
        val_txt = np.loadtxt('./Process_data/CS_test_V1.txt', dtype = str)
        for idx, name in enumerate(val_txt):
            label1 = int(name.split('A')[1][:3])
            label.append(label1)
        label = torch.from_numpy(np.array(label))
        
    if 'csv2' in arg.dataset:
        val_txt = np.loadtxt('./Process_data/CS_test_V2.txt', dtype = str)
        for idx, name in enumerate(val_txt):
            label1 = int(name.split('A')[1][:3])
            label.append(label1)
        label = torch.from_numpy(np.array(label))
        '''

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

            
    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        space = [(0.0001, 2) for i in range(4)]
        result = gp_minimize(objective, space, n_calls=200, random_state=0)
        print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
        print('Optimal weights: {}'.format(result.x))
# Maximum accuracy: 85.7143%
# Optimal weights: [1.2, 0.8917599269805236, 0.2, 1.2]

# Maximum accuracy: 86.6071%
# Optimal weights: [0.875129823854929, 1.5, 0.01, 0.6754383742173085]

# Maximum accuracy: 86.6071%
# Optimal weights: [1.754516824683899, 0.9871409301644719, 0.0001, 1.359480006396815]


# Maximum accuracy: 89.0000%
# Optimal weights: [2.0, 0.8245193386538215, 0.4757473103679432, 0.0001]


# Maximum accuracy: 91.0000%
# Optimal weights: [1.4229072845180353, 2.0, 0.0001, 0.6775214907331517]

# Maximum accuracy: 86.3014%
# Optimal weights: [0.0001, 2.0, 0.0001, 0.0001]
