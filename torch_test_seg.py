import numpy as np
import torch_provider
import torch_Models
import torch
# import torch_tensorrt

import os
import torch.nn as nn
import torch.utils.data
from torch.autograd.variable import Variable
import time


def main():
    # path and something...
    model_name = 'PointNet_Seg'
    model_save_path = 'checkpoints/'
    num_points = 1024
    num_classes = 50
    # -----
    lr = 0.001
    batch_size = 1
    epochs = 250

    # load
    print('*******************************************')
    print('loading data...')
    TRAINING_FILE_LIST = os.path.join('data/hdf5_data', 'train_hdf5_file_list.txt')
    TESTING_FILE_LIST = os.path.join('data/hdf5_data', 'val_hdf5_file_list.txt')
    train_file_list = torch_provider.getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)
    test_file_list = torch_provider.getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)

    train_data = np.zeros([0, num_points, 3])
    train_seg = np.zeros([0, num_points])
    train_label = np.zeros([0])
    train_file_idxs = np.arange(0, num_train_file)
    np.random.shuffle(train_file_idxs)
    for i in range(num_train_file):
        cur_train_filename = os.path.join('data/hdf5_data', train_file_list[train_file_idxs[i]])
        cur_data, cur_labels, cur_seg = torch_provider.loadDataFile_with_seg(cur_train_filename)
        cur_data, cur_labels, order = torch_provider.shuffle_data(cur_data, np.squeeze(cur_labels))
        cur_seg = cur_seg[order, ...]
        cur_data = cur_data[:, 0:num_points, :]
        cur_seg = cur_seg[:, 0:num_points]
        train_data = np.concatenate((train_data, cur_data), axis=0)
        train_label = np.concatenate((train_label, cur_labels), axis=0)
        train_seg = np.concatenate((train_seg, cur_seg), axis=0)
    print(train_data.shape)
    print(train_seg.shape)
    print(train_label.shape)

    test_data = np.zeros([0, num_points, 3])
    test_seg = np.zeros([0, num_points])
    test_label = np.zeros([0])
    for i in range(num_test_file):
        cur_test_filename = os.path.join('data/hdf5_data', test_file_list[i])
        cur_data, cur_labels, cur_seg = torch_provider.loadDataFile_with_seg(cur_test_filename)
        cur_data, cur_labels, order = torch_provider.shuffle_data(cur_data, np.squeeze(cur_labels))
        cur_seg = cur_seg[order, ...]
        cur_data = cur_data[:, 0:num_points, :]
        cur_seg = cur_seg[:, 0:num_points]
        test_data = np.concatenate((test_data, cur_data), axis=0)
        test_label = np.concatenate((test_label, cur_labels), axis=0)
        test_seg = np.concatenate((test_seg, cur_seg), axis=0)
    print(test_data.shape)
    print(test_seg.shape)
    print(test_label.shape)

    train_data_ = []
    valid_data_ = []
    for i in range(0, int(0.9 * len(train_data))):
        # train_seg_ = to_categorical(int(train_seg[i]), num_classes)
        train_data_.append([train_data[i], train_seg[i]])
    for i in range(int(0.9 * len(train_data)), len(train_data)):
        valid_data_.append([train_data[i], train_seg[i]])
    test_data_ = []
    for i in range(len(test_data)):
        # test_label_ = to_categorical(int(test_label[i]), num_classes)
        test_data_.append([test_data[i], test_seg[i]])


    train_loader = torch.utils.data.DataLoader(train_data_,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               shuffle=True,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data_,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               shuffle=False,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data_,
                                              batch_size=batch_size,
                                              num_workers=8,
                                              shuffle=False,
                                              pin_memory=True)
    print('batch num:')
    print('train dataset: ' + str(len(train_loader)))
    print('valid dataset: ' + str(len(valid_loader)))
    print('test dataset: ' + str(len(test_loader)))

    # model
    print('*******************************************')
    print('building model...')
    model_seg = torch_Models.pointnet_seg(num_classes).cuda()
    model_seg_data = torch.load(os.path.join(model_save_path, 'pc_seg.pth'))
    model_seg.load_state_dict(model_seg_data)
    # params = get_n_params(model_cls)
    # print('Number of parameters: ' + str(params) + '...\n')

    acc = evaluate(test_loader, model_seg)
    print("Acc: ", acc)


def evaluate(valid_loader, model):
    model.eval()

    num_correct = 0
    time_ = []
    time_2 = []
    for i, (input, target) in enumerate(valid_loader):
        with torch.no_grad():

            start = time.time()
            input = input.cuda(non_blocking=True)
            input_var = Variable(input, requires_grad=True)
            trans_input = torch.transpose(input_var, 1, 2).type(torch.FloatTensor).cuda()
            # pred_cls, trans_feat = model(trans_input)
            pred_cls = model(trans_input)
            time_.append(time.time() - start)
            target = target.type(torch.LongTensor).cuda()

            for j in range(pred_cls.shape[0]):
                pred_cls_ = torch.argmax(pred_cls[j, :, :], dim=1)
                num_correct += torch.eq(pred_cls_, target[j, :]).sum().float().item()

    print("Avg Processing Time:",round(np.mean(time_[1:])*1000, 2), "ms")

    return num_correct/(1870*1024)

if __name__ == "__main__":
    main()