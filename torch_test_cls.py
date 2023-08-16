import numpy as np
import torch_provider
import torch_Models
import torch
import os
import torch.utils.data
from torch.autograd.variable import Variable

def main():
    # path and something...
    model_save_path = 'checkpoints/'
    num_points = 1024
    num_classes = 40
    # -----
    lr = 0.001
    batch_size = 32
    epochs = 250

    # load
    print('*******************************************')
    print('loading data...')
    TRAIN_FILES = torch_provider.getDataFiles('data/modelnet40_ply_hdf5_2048/train_files.txt')
    TEST_FILES = torch_provider.getDataFiles('data/modelnet40_ply_hdf5_2048/test_files.txt')

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    train_data = np.zeros([0, num_points, 3])
    train_label = np.zeros([0])
    for fn in range(len(TRAIN_FILES)):
        current_data, current_label = torch_provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:num_points,:]
        current_data, current_label, _ = torch_provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)
        train_data = np.concatenate((train_data, current_data), axis=0)
        train_label = np.concatenate((train_label, current_label), axis=0)
    # rotate jitter
    # train_data = torch_provider.rotate_point_cloud(train_data)
    # train_data = torch_provider.jitter_point_cloud(train_data)
    print(np.array(train_data).shape)
    print(np.array(train_label).shape)

    test_data = np.zeros([0, num_points, 3])
    test_label = np.zeros([0])
    for fn in range(len(TEST_FILES)):
        current_data, current_label = torch_provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:num_points,:]
        current_label = np.squeeze(current_label)
        test_data = np.concatenate((test_data, current_data), axis=0)
        test_label = np.concatenate((test_label, current_label), axis=0)
    print(np.array(test_data).shape)
    print(np.array(test_label).shape)



    # one-hot
    # train_label = to_categorical(int(train_label), num_classes)
    # test_label = to_categorical(int(test_label), num_classes)
    # print(train_label.shape)
    # print(test_label.shape)

    train_data_ = []
    valid_data_ = []
    for i in range(0, int(0.9 * len(train_data))):
        # train_label_ = to_categorical(int(train_label[i]), num_classes)
        train_data_.append([train_data[i], train_label[i]])
    for i in range(int(0.9 * len(train_data)), len(train_data)):
        valid_data_.append([train_data[i], train_label[i]])
    test_data_ = []
    for i in range(len(test_data)):
        # test_label_ = to_categorical(int(test_label[i]), num_classes)
        test_data_.append([test_data[i], test_label[i]])


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
    model_cls = torch_Models.pointnet_cls(num_classes).cuda()
    model_cls_data = torch.load(os.path.join(model_save_path, 'pc_cls.pth'))
    model_cls.load_state_dict(model_cls_data)
    # params = get_n_params(model_cls)
    # print('Number of parameters: ' + str(params) + '...\n')

    acc = evaluate(test_loader, model_cls)
    print(acc)


def evaluate(valid_loader, model):
    model.eval()

    num_correct = 0
    for i, (input, target) in enumerate(valid_loader):
        with torch.no_grad():

            input = input.cuda(non_blocking=True)
            input_var = Variable(input, requires_grad=True)
            trans_input = torch.transpose(input_var, 1, 2).type(torch.FloatTensor).cuda()
            target = target.type(torch.LongTensor).cuda()
            pred_cls = model(trans_input)
            pred_cls = torch.argmax(pred_cls, dim=1)

            num_correct += torch.eq(pred_cls, target).sum().float().item()

    return num_correct/2468


if __name__ == "__main__":
    main()