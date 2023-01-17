mkdir -p checkpoints_and_pseudo_labels/checkpoints
mkdir -p checkpoints_and_pseudo_labels/pseudo_labels

wget -O checkpoints_and_pseudo_labels/checkpoints/self_training_init.pth 'https://www.dropbox.com/s/o9ahjk84jvit6lo/self_training_init.pth?dl=0'
wget -O checkpoints_and_pseudo_labels/checkpoints/self_training_aug_a.pth 'https://www.dropbox.com/s/ggssct6uupfmrxw/Aug_A.pth?dl=0'
wget -O checkpoints_and_pseudo_labels/checkpoints/self_training_aug_b.pth 'https://www.dropbox.com/s/vtlggg222cwro1d/Aug_B.pth?dl=0'
wget -O checkpoints_and_pseudo_labels/checkpoints/self_training_aug_c.pth 'https://www.dropbox.com/s/ujtoe3ec9ci7p4h/Aug_C.pth?dl=0'
wget -O checkpoints_and_pseudo_labels/checkpoints/model_soup.pth 'https://www.dropbox.com/s/8yodretu0mldybf/model_soup.pth?dl=0'
wget -O checkpoints_and_pseudo_labels/pseudo_labels/ 'https://www.dropbox.com/s/faugl9zwicgavgm/Pseudo_Labels.zip?dl=0'