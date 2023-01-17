# Default HRDA Configuration for GTA->Cityscapes
_base_ = [
    '../../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../../_base_/models/_visda2022/_visda2022_convnext_latest.py',
    # GTA->Cityscapes High-Resolution Data Loading
    '../../_base_/datasets/_visda2022/_uda/uda_zerowastev1_to_zerowastev2.py',
    # DAFormer Self-Training
    '../../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 1

uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[1, 2, 3, 4],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120)
   
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=1)
evaluation = dict(interval=40000, metric='mIoU')
# Meta Information for Result Analysis
name = '_visda2022_zerowastev1_to_zerowastev2_HDRA_baseline'
exp = 'basic'
name_dataset = '_visda2022_zerowastev1_to_zerowastev2'
name_architecture = 'hrda1-512-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-512-0.1_daformer_sepaspp_sl'
name_uda = 'dacs_a999_fdthings_rcs0.01-2.0_cpl2'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

# For the other configurations used in the paper, please refer to experiment.py