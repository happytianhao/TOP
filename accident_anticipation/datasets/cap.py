# Copyright (c) OpenMMLab. All rights reserved.
import os
import pandas as pd

from mmaction.registry import DATASETS
from mmengine.dataset import Compose, BaseDataset


@DATASETS.register_module()
class CAPDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        ann_file,
        filename_tmpl="{:06}.jpg",
        start_index=1,
        pipeline=None,
        test_mode=False,
        indices=None,
    ):
        self.data_root = data_root
        self.ann_file = ann_file
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        self.test_mode = test_mode
        self._indices = indices
        self._metainfo = self._load_metainfo(None)
        self.serialize_data = True
        self.max_refetch = 1000
        self.pipeline = Compose(pipeline)
        self.full_init()

    def load_data_list(self):
        data_list = []

        df = pd.read_csv(os.path.join(self.data_root, self.ann_file))

        # 遍历每一行数据
        for _, row in df.iterrows():
            data_info = {
                "dataset": "cap",
                "filename": None,
                "frame_dir": os.path.join(self.data_root, row["path"]),
                "filename_tmpl": self.filename_tmpl,
                "start_index": self.start_index,
                "video_id": str(row["video_id"]).zfill(6),  # 转换为六位字符串格式
                "type": row["type"],
                "have_accident": bool(row["have_accident"]),
                "abnormal_start_frame": row["abnormal_start_frame"],
                "accident_frame": row["accident_occur_frame"],
                "total_frames": row["total_frames"],
                "fps": row["fps"],
            }
            data_list.append(data_info)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info["modality"] = "RGB"
        return data_info
