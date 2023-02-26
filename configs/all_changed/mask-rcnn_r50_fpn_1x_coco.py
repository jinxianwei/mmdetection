_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

train_dataloader = dict(batch_size=4)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=3)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR', by_epoch=True, T_max=100, eta_min=1e-6)

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'

work_dir = '/root/autodl-tmp/mmdet_workdir/mask-rcnn'