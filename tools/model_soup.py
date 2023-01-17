import os
import os.path
import argparse

import torch



def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument('--weights_init', default='', type=str)
    parser.add_argument('--weights_a', default='', type=str)
    parser.add_argument('--weights_b', default='', type=str)
    parser.add_argument('--weights_c', default='', type=str)
    parser.add_argument('--save_weights', default='', type=str)

    args = parser.parse_args()
    return args


def process(checkpoint_base, checkpoint_weights1, checkpoint_weights2, checkpoint_weights3, save_weights):

    state_dict_base = checkpoint_base['state_dict']
    state_dict_weights1 = checkpoint_weights1['state_dict']
    state_dict_weights2 = checkpoint_weights2['state_dict']
    state_dict_weights3 = checkpoint_weights3['state_dict']

    save_state_dict = {"state_dict": dict(), "meta": dict()}

    save_state_dict["meta"]["CLASSES"] = [
        "background",
        "rigid_plastic",
        "cardboard",
        "metal",
        "soft_plastic",
    ]
    
    save_state_dict["meta"]["PALETTE"] = [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
    ]

    for (n, p1), (_, p2), (_, p3), (_, p4) in zip(state_dict_base.items(), state_dict_weights1.items(), state_dict_weights2.items(), state_dict_weights3.items()):

        save_state_dict["state_dict"][n] = (p1 + p2 + p3 + p4) / 4.0

    torch.save(save_state_dict, save_weights)


def main():
    args = parse_args()


    checkpoint_base = torch.load(args.weights_init)
    checkpoint_weights1 = torch.load(args.weights_a)
    checkpoint_weights2 = torch.load(args.weights_b)
    checkpoint_weights3 = torch.load(args.weights_c)

    process(checkpoint_base = checkpoint_base,
            checkpoint_weights1 = checkpoint_weights1,
            checkpoint_weights2 = checkpoint_weights2,
            checkpoint_weights3 = checkpoint_weights3,
            save_weights = args.save_weights)


if __name__ == "__main__":
    main()

