_base_ = ["./top_cap.py"]

# 仅覆盖数据集与评估器，其他配置继承自 top_cap

train_dataloader = dict(
    dataset=dict(
        type="DADADataset",
        data_root="data/MM-AU/DADA-DATA",
        ann_file="dada_train_annotations.csv",
        filename_tmpl="{:04}.png",
        start_index=1,
    )
)

val_dataloader = dict(
    dataset=dict(
        type="DADADataset",
        data_root="data/MM-AU/DADA-DATA",
        ann_file="dada_val_annotations.csv",
        filename_tmpl="{:04}.png",
        start_index=1,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    data_root="data/MM-AU/DADA-DATA",
    ref_file="dada_val_references.csv",
)

test_evaluator = val_evaluator
