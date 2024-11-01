_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/turtle.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# 修改模型配置
model = dict(
    decode_head=dict(num_classes=4),  # 类别数量（包括背景）
    auxiliary_head=dict(num_classes=4)
)

# 训练配置
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=400)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)

# 学习率配置
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False
    )
]