import os
import pandas as pd

from mmaction.registry import DATASETS
from mmengine.dataset import Compose, BaseDataset


@DATASETS.register_module()
class NexarDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        ann_file,
        filename_tmpl="{:06}.jpg",
        start_index=0,
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

        for _, row in df.iterrows():
            # 输入表头: video_id,path,fps,total_frames,abnormal_start_frame,accident_occur_frame,have_accident
            video_id = str(row["video_id"]).zfill(5)
            have_accident = int(row["have_accident"])
            accident_frame = int(row["accident_occur_frame"])
            abnormal_start_frame = int(row["abnormal_start_frame"])
            data_info = {
                "dataset": "nexar",
                "filename": None,
                "frame_dir": os.path.join(self.data_root, row["path"]),
                "filename_tmpl": self.filename_tmpl,
                "start_index": self.start_index,
                "video_id": video_id,
                "type": None,
                # 将异常/事故帧的 -1 规范为 None
                "accident_frame": None if accident_frame == -1 else accident_frame,
                "abnormal_start_frame": None if abnormal_start_frame == -1 else abnormal_start_frame,
                "total_frames": row["total_frames"],
                "fps": row["fps"],
                # have_accident: -1 -> None, 0/1 -> False/True
                "have_accident": None if have_accident == -1 else bool(have_accident),
            }
            data_list.append(data_info)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info["modality"] = "RGB"
        return data_info
