_base_ = ["./top_cap.py"]

# 仅覆盖数据集与评估器，其他配置继承自 top_cap

train_dataloader = dict(
    dataset=dict(
        type="NexarDataset",
        data_root="data/nexar-collision-prediction",
        ann_file="nexar_train_annotations.csv",
        filename_tmpl="{:06}.jpg",
        start_index=0,
    )
)

val_dataloader = dict(
    dataset=dict(
        type="NexarDataset",
        data_root="data/nexar-collision-prediction",
        ann_file="nexar_val_annotations.csv",
        filename_tmpl="{:06}.jpg",
        start_index=0,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    data_root="data/nexar-collision-prediction",
    ref_file="nexar_val_references.csv",
)

test_evaluator = val_evaluator
