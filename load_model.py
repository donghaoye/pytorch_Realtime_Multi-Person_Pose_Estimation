import torch.nn as nn

def load_model():

    blocks = {}

    block0 = [{'conv1_1': [3, 64, 3, 1, 1]}, {'conv1_2': [64, 64, 3, 1, 1]}, {'pool1_stage1': [2, 2, 0]},
              {'conv2_1': [64, 128, 3, 1, 1]}, {'conv2_2': [128, 128, 3, 1, 1]}, {'pool2_stage1': [2, 2, 0]},
              {'conv3_1': [128, 256, 3, 1, 1]}, {'conv3_2': [256, 256, 3, 1, 1]}, {'conv3_3': [256, 256, 3, 1, 1]},
              {'conv3_4': [256, 256, 3, 1, 1]}, {'pool3_stage1': [2, 2, 0]}, {'conv4_1': [256, 512, 3, 1, 1]},
              {'conv4_2': [512, 512, 3, 1, 1]}, {'conv4_3_CPM': [512, 256, 3, 1, 1]},
              {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    for i in range(2, 7):
        blocks['block%d_1' % i] = [{'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
                                   {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                                   {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}]
        blocks['block%d_2' % i] = [{'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
                                   {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                                   {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}]

    def make_layers(cfg_dict):
        layers = []
        for i in range(len(cfg_dict) - 1):
            one_ = cfg_dict[i]
            for k, v in one_.iteritems():
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]
        one_ = cfg_dict[-1].keys()
        k = one_[0]
        v = cfg_dict[-1][k]
        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        layers += [conv2d]
        return nn.Sequential(*layers)

    layers = []
    for i in range(len(block0)):
        one_ = block0[i]
        for k, v in one_.iteritems():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]

    models = {}
    models['block0'] = nn.Sequential(*layers)

    for k, v in blocks.iteritems():
        models[k] = make_layers(v)

    return models