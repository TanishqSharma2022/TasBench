_base_ = [
    '../../_base_/default_runtime.py'
]


alphabet_file = '/home/tanish/scratch/tanishq/table-understanding/repos/PIX-TAB/dict_papp_otsl.txt' # Changed the vocabulary
alphabet_len = len(open(alphabet_file, 'r').readlines())
max_seq_len = 1024 # longer due to row and col tokens being added

# cell_alphabet_file = './tools/data/alphabet/pubtabnet/textline_recognition_alphabet.txt'
# cell_alphabet_len = len(open(cell_alphabet_file, 'r').readlines())
# max_seq_len_cell = 150

start_end_same = False
label_convertor = dict(
            type='TableMasterConvertor',
            dict_file=alphabet_file,
            max_seq_len=max_seq_len,
            start_end_same=start_end_same,
            with_unknown=True,
            # cell_dict_file=cell_alphabet_file,
            # max_seq_len_cell=max_seq_len_cell
            )

if start_end_same:
    PAD = alphabet_len + 2
else:
    PAD = alphabet_len + 3

# # cell content
# if start_end_same:
#     PAD_CELL = cell_alphabet_len + 2
# else:
#     PAD_CELL = cell_alphabet_len + 3

model = dict(
    type='TABLEMASTER',
    backbone=dict(
        type='TableResNetExtra',
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True],
        ),
        layers=[1,2,5,3]),
    encoder=dict(
        type='PositionalEncoding',
        d_model=512,
        dropout=0.2,
        max_len=5000),
    decoder=dict(
        type='TableMasterDecoder',
        N=3,
        decoder=dict(
            self_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            src_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            feed_forward=dict(
                d_model=512,
                d_ff=2024,
                dropout=0.),
            size=512,
            dropout=0.),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=PAD, reduction='mean'),
    bbox_loss=dict(type='TableL1Loss', reduction='sum'),
    # cell_loss=dict(type='MASTERCELLLoss', ignore_inderx=PAD_CELL, reduction='mean'),
    cell_loss=None,
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)


TRAIN_STATE = True
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'scale_factor',
            'bbox', 'bbox_masks', 'pad_shape' 
            # 'cell_content'
        ]),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename', 'bbox', 'bbox_masks', 'pad_shape'
            # , 'cell_content'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    # dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename', 'pad_shape'
            # 'filename', 'ori_shape', 'img_shape', 'scale_factor',
            # 'img_norm_cfg', 'ori_filename', 'bbox', 'bbox_masks', 'pad_shape', 'cell_content'
        ]),
]

dataset_type = 'PubTabNetPAPPOTSL'
# train_img_prefix = '/home/sarthak/scratch/tanishq/table-understanding/data/Table Datasets/PubTabNet_PIXTAB_corrected/images'
# train_anno_file1 = '/home/sarthak/scratch/tanishq/table-understanding/data/Table Datasets/PubTabNet_PIXTAB_corrected/train_pix_tab_cleaned.jsonl'

train_img_prefix = '/home/sarthak/scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/pixtab/images'
train_anno_file1 = '/home/sarthak/scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/pixtab/train_pix_tab.jsonl'


train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix,
    ann_file=train_anno_file1,
    pipeline=train_pipeline,
    test_mode=False,
    filter_empty_gt=False
    )

valid_img_prefix = '/home/sarthak/scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/pixtab/images'
valid_anno_file1 = '/home/sarthak/scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/pixtab/val_pix_tab.jsonl'
valid = dict(
    type=dataset_type,
    img_prefix=valid_img_prefix,
    ann_file=valid_anno_file1,
    pipeline=valid_pipeline,
    test_mode=True)

test_img_prefix = '/home/sarthak/scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/pixtab/images'
test_anno_file1 = '/home/sarthak/scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/pixtab/val_pix_tab.jsonl'
test = dict(
    type=dataset_type,
    img_prefix=test_img_prefix,
    ann_file=test_anno_file1,
    pipeline=test_pipeline,
    test_mode=True)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(type='ConcatDataset', datasets=[train1]),
    val=dict(type='ConcatDataset', datasets=[valid]),
    test=dict(type='ConcatDataset', datasets=[test], samples_per_gpu=1))

# optimizer
optimizer = dict(type='Ranger', lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(
    # cumulative_iters=4,                          # Accumulate over N steps
    grad_clip=dict(max_norm=35, norm_type=2)
)
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
#    step=[12, 15])
    step=[20, 28])
total_epochs = 50

# evaluation
# evaluation = dict(interval=1, metric='acc')

# fp16
# fp16 = dict(loss_scale='dynamic')
fp16 = None

# checkpoint setting
checkpoint_config = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# log_config
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')

    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/home/sarthak/scratch/tanishq/table-understanding/repos/PIX-TAB/full-pubtabnet-500k/latest.pth'
# resume_from = None
# resume_from = '/home/sarthak/scratch/tanishq/table-understanding/repos/PIX-TAB/full-pubtabnet-500k/latest.pth'
workflow = [('train', 1)]

# if raise find unused_parameters, use this.
find_unused_parameters = True