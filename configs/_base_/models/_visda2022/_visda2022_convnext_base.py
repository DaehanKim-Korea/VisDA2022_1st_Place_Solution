# This is the same as SegFormer but with 256 embed_dims
# SegF. with C_e=256 in Tab. 7

custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

model_size = "large"

if model_size == 'large': # 200M / 300M
    arch_cfg = 'large'
    in_channels_cfg = [192, 384, 768, 1536]
    checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'
    channels_cfg = 512

else:
    arch_cfg = 'base' # 80M / 300M
    in_channels_cfg=[128, 256, 512, 1024]
    checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'  # noqa
    channels_cfg = 256

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
    type='mmcls.ConvNeXt',
    arch=arch_cfg,

    out_indices=[0, 1, 2, 3],
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    gap_before_final_norm=False,
    init_cfg=dict(
        type='Pretrained', checkpoint=checkpoint_file,
        prefix='backbone.')),
        
    decode_head=dict(
        type='DAFormerHead',
        in_channels=in_channels_cfg,
        in_index=[0, 1, 2, 3],
        channels=channels_cfg,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='conv',
                kernel_size=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg),
        ),

        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
