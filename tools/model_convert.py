import os
import os.path
import argparse

import torch
from collections import OrderedDict
def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch')
    parser.add_argument('--load_weights', default='', type=str)
    parser.add_argument('--save_weights', default='', type=str)

    args = parser.parse_args()
    return args


def process(load_weights, save_weights):

    checkpoint_base = torch.load(load_weights)
    state_dict_base = checkpoint_base['state_dict']

    save_state_dict = OrderedDict()
    save_state_dict['meta'] = OrderedDict() #'meta': dict(), 'state_dict': dict()}
    save_state_dict['state_dict'] = OrderedDict() #'meta': dict(), 'state_dict': dict()}

    save_state_dict['meta']['CLASSES'] = ["background", "rigid_plastic", "cardboard", "metal", "soft_plastic"]
    save_state_dict['meta']['PALETTE'] = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156]]

    for (n, p) in (state_dict_base.items()):

        if not ((n.split('.')[0] == 'ema_model') or (n.split('.')[0] == 'imnet_model')):
            n = '.'.join(n.split('.')[1:])
            save_state_dict['state_dict'][n] = p

    torch.save(save_state_dict, save_weights)



def main():
    args = parse_args()
    process(args.load_weights, args.save_weights)


if __name__ == '__main__':
    main()
