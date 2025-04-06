_base_ = ["_base_/schedules/sgd_50e.py", "_base_/default_runtime.py"]

custom_imports = dict(imports="taa")

# dataset settings
dataset_type = "NexarCollisionPrediction"
data_root = "data/nexar-collision-prediction"
ann_file = "annotations.csv"
filename_tmpl = "{:06}.jpg"
vis_list = [
    "00073",
    "00080",
    "00150",
    "00178",
    "00196",
    "00293",
    "00322",
    "00421",
    "00445",
    "00446",
    "00468",
    "00595",
    "00619",
    "00797",
    "00807",
    "00826",
    "00870",
    "00892",
    "00894",
    "00951",
    "01015",
    "01032",
    "01038",
    "01059",
    "01143",
    "01161",
    "01169",
    "01241",
    "01275",
    "01351",
    "01353",
    "01373",
    "01428",
    "01515",
    "01556",
    "01619",
    "01648",
    "01650",
    "01675",
    "01703",
    "01794",
    "01814",
    "01854",
    "01868",
    "01927",
    "02043",
    "02086",
    "02127",
    "02141",
    "02149",
    "02153",
    "02171",
    "02179",
    "02181",
    "02195",
    "02196",
    "02228",
    "02243",
    "02247",
    "02257",
    "02269",
    "02298",
    "02321",
    "02336",
    "02340",
    "02343",
    "02363",
    "02372",
    "02389",
    "02395",
    "02402",
    "02406",
    "02410",
    "02416",
    "02417",
    "02419",
    "02443",
    "02445",
    "02450",
    "02506",
    "02516",
    "02584",
    "02667",
    "02675",
    "02679",
    "02681",
    "02724",
    "02736",
    "02747",
    "02754",
    "02755",
    "02760",
    "02788",
    "02801",
    "02807",
    "02810",
    "02815",
    "02823",
    "02881",
    "02914",
]

algorithm_keys = (
    "frame_dir",
    "filename_tmpl",
    "img_shape",
    "sample_idx",
    "video_id",
    "label",
    "start_index",
    "total_frames",
    "target",
    "abnormal_start_frame",
    "accident_frame",
    "frame_inds",
    "clip_len",
    "num_clips",
    "frame_interval",
    "fps",
    "is_val",
    "is_test",
)

file_client_args = dict(io_backend="disk")

train_pipeline = [
    dict(type="SampleSnippetsForNexar", snippet_len=5, num_snippets=10, test_mode=False),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="RandomResizedCrop", area_range=(0.6, 1.0), aspect_ratio_range=(1.0, 16 / 9)),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
    # dict(type="VisualizeInputsAsVideos", output_dir="visualizations/inputs_train"),
]
val_pipeline = [
    dict(type="SampleSnippetsForNexar", snippet_len=5, num_snippets=60, test_mode=True),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    # dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        filename_tmpl=filename_tmpl,
        pipeline=train_pipeline,
        test_mode=False,
        train_with_val=False,
        # indices=list(range(20)),
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        filename_tmpl=filename_tmpl,
        pipeline=val_pipeline,
        test_mode=True,
        # indices=list(range(20)),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="NexarMetric",
    vis_list=vis_list,
    output_dir="visualizations",
)
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_begin=1, val_interval=1)

# 每轮都保存权重，并且只保留最新的权重
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=50, save_best="mAP_val", rule="greater")
)
custom_hooks = [dict(type="EpochHook"), dict(type="NexarMetricHook")]

model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="ResNet3dSlowOnly",
        depth=50,
        pretrained="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False,
    ),
    cls_head=dict(
        type="OccurrenceHeadFromSnippetsForNexar",
        num_classes=1,
        num_decoder_layers=2,
        loss_cls=dict(type="BCELossWithLogits"),
        pos_weight=10,
        observed_len=5,
        anticipated_len=25,
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], format_shape="NCTHW"
    ),
    train_cfg=None,
    test_cfg=None,
)

# load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb_20230612-12ce977c.pth"
load_from = "epoch_39_0.0567_0.8063_0.6485.pth"
# load_from = "epoch_6_0.0251_0.8287_0.6374.pth"
