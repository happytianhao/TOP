# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Union

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from mmengine.dataset import Compose, BaseDataset
from mmaction.datasets import BaseActionDataset

import pandas as pd
import numpy as np

from .utils import get_fps
from .splits import cap_test, dada_test, nexar_val


@DATASETS.register_module()
class CapData(BaseActionDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video.


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where video
            frames are held. Defaults to ``dict(img='')``.
        filename_tmpl (str): Template for each filename.
            Defaults to ``img_{:05}.jpg``.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Defaults to False.
        num_classes (int, optional): Number of classes in the dataset.
            Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking frames as input,
            it should be set to 1, since raw frames count from 1.
            Defaults to 1.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``RGB``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(
        self,
        ann_file: str,
        pipeline: List[Union[ConfigType, Callable]],
        data_prefix: ConfigType = dict(img=""),
        filename_tmpl: str = "img_{:05}.jpg",
        multi_class: bool = False,
        num_classes: Optional[int] = None,
        start_index: int = 1,
        modality: str = "RGB",
        test_mode: bool = False,
        val_train: bool = False,
        **kwargs,
    ) -> None:
        self.filename_tmpl = filename_tmpl
        self.val_train = val_train
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = pd.read_excel(self.ann_file, sheet_name=1).values.tolist()
        for line in fin:
            """
            video_id, weather, light, scenes, linear, type, have accident,
            abnormal start frame, abnormal end frame, accident frame, total frames,
            [0,tai], [tai,tco], [tai,tae], [tco,tae], [tae,end],
            texts, causes, measures
            """

            video_id = str(line[0]).zfill(6)
            # 1-10, 11, 12-42, 43, 44-62
            if 1 <= line[5] <= 10:
                frame_dir = "1-10"
            elif line[5] == 11:
                frame_dir = "11"
            elif 12 <= line[5] <= 42:
                frame_dir = "12-42"
            elif line[5] == 43:
                frame_dir = "43"
            elif 44 <= line[5] <= 62:
                frame_dir = "44-62"
            frame_dir = osp.join(self.data_prefix["img"], frame_dir, str(line[5]), video_id, "images")

            # skip the broken videos
            if video_id in ["011665", "008728", "004928", "007155"]:
                continue

            # only for ego-car accidents
            if not 1 <= line[5] <= 18:
                continue

            # keep the train videos
            if not self.test_mode and video_id in cap_test.keys():
                continue

            # keep the test videos
            if self.test_mode and not self.val_train and video_id not in cap_test.keys():
                continue

            correct_total_frames = {
                "005722": 78,
                "009073": 211,
                "009074": 220,
                "009480": 586,
                "010087": 301,
                "009541": 453,
                "009222": 303,
                "009255": 217,
                "008589": 106,
            }

            # fix the total frames
            if video_id in correct_total_frames.keys():
                line[10] = correct_total_frames[video_id]

            data_list.append(
                dict(
                    frame_dir=frame_dir,
                    video_id=video_id,
                    weather=line[1],
                    light=line[2],
                    scenes=line[3],
                    linear=line[4],
                    type=line[5],
                    have_accident=line[6],
                    abnormal_start_frame=line[7],
                    abnormal_end_frame=line[8],
                    accident_frame=line[9],
                    total_frames=line[10],
                    start2tai=line[11],
                    tai2tco=line[12],
                    tai2tae=line[13],
                    tco2tae=line[14],
                    tae2end=line[15],
                    texts=line[16],
                    causes=line[17],
                    measures=line[18],
                    fps=get_fps(video_id, dataset_type="cap"),
                    is_test=video_id in cap_test.keys(),
                )
            )
        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info["filename_tmpl"] = self.filename_tmpl
        return data_info


@DATASETS.register_module()
class DadaData(BaseActionDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video.


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where video
            frames are held. Defaults to ``dict(img='')``.
        filename_tmpl (str): Template for each filename.
            Defaults to ``img_{:05}.jpg``.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Defaults to False.
        num_classes (int, optional): Number of classes in the dataset.
            Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking frames as input,
            it should be set to 1, since raw frames count from 1.
            Defaults to 1.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``RGB``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(
        self,
        ann_file: str,
        pipeline: List[Union[ConfigType, Callable]],
        data_prefix: ConfigType = dict(img=""),
        filename_tmpl: str = "img_{:05}.jpg",
        multi_class: bool = False,
        num_classes: Optional[int] = None,
        start_index: int = 1,
        modality: str = "RGB",
        test_mode: bool = False,
        val_train: bool = False,
        **kwargs,
    ) -> None:
        self.filename_tmpl = filename_tmpl
        self.val_train = val_train
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = pd.read_excel(self.ann_file, sheet_name=1).values.tolist()
        for line in fin:
            """
            video_id, weather, light, scenes, linear, type, have accident,
            abnormal start frame, abnormal end frame, accident frame, total frames,
            [0,tai], [tai,tco], [tai,tae], [tco,tae], [tae,end],
            texts, causes, measures
            """

            video_id = str(line[0]).zfill(3)
            frame_dir = osp.join(self.data_prefix["img"], str(line[5]), video_id, "images")
            video_id = f"{line[5]}_{video_id}"

            # skip the videos without accidents
            if line[8] == -1:
                continue

            # only for ego-car accidents
            if not 1 <= line[5] <= 18:
                continue

            # keep the train videos
            if not self.test_mode and video_id in dada_test.keys():
                continue

            # keep the test videos
            if self.test_mode and not self.val_train and video_id not in dada_test.keys():
                continue

            correct_total_frames = {
                "5_040": 382,
                "5_049": 204,
                "6_009": 325,
                "10_169": 320,
                "11_076": 666,
                "11_139": 220,
                "36_002": 342,
                "37_003": 695,
                "43_080": 338,
                "43_188": 220,
                "48_065": 330,
                "50_136": 422,
                "56_008": 433,
            }

            # fix the total frames
            if video_id in correct_total_frames.keys():
                line[10] = correct_total_frames[video_id]

            data_list.append(
                dict(
                    frame_dir=frame_dir,
                    video_id=video_id,
                    weather=line[1],
                    light=line[2],
                    scenes=line[3],
                    linear=line[4],
                    type=line[5],
                    have_accident=line[6],
                    abnormal_start_frame=line[7],
                    accident_frame=line[8],
                    abnormal_end_frame=line[9],
                    total_frames=line[10],
                    start2tai=line[11],
                    tai2tco=line[12],
                    tai2tae=line[13],
                    tco2tae=line[14],
                    tae2end=line[15],
                    texts=line[16],
                    causes=line[17],
                    measures=line[18],
                    fps=get_fps(video_id, dataset_type="dada"),
                    is_test=video_id in dada_test.keys(),
                )
            )
        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info["filename_tmpl"] = self.filename_tmpl
        return data_info


@DATASETS.register_module()
class NexarCollisionPrediction(BaseActionDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video.


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where video
            frames are held. Defaults to ``dict(img='')``.
        filename_tmpl (str): Template for each filename.
            Defaults to ``img_{:05}.jpg``.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Defaults to False.
        num_classes (int, optional): Number of classes in the dataset.
            Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking frames as input,
            it should be set to 1, since raw frames count from 1.
            Defaults to 1.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``RGB``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(
        self,
        ann_file: str,
        pipeline: List[Union[ConfigType, Callable]],
        data_prefix: ConfigType = dict(img=""),
        filename_tmpl: str = "img_{:05}.jpg",
        multi_class: bool = False,
        num_classes: Optional[int] = None,
        start_index: int = 0,
        modality: str = "RGB",
        test_mode: bool = False,
        train_with_val: bool = True,
        **kwargs,
    ) -> None:
        self.filename_tmpl = filename_tmpl
        self.train_with_val = train_with_val
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = pd.read_csv(self.ann_file).values.tolist()
        for line in fin:
            video_id = str(int(line[0])).zfill(5)
            is_test = bool(line[1])
            target = bool(line[5]) if not np.isnan(line[5]) else None
            fps = 30
            if not is_test:
                frame_dir = "train_raw_frames"
            else:
                frame_dir = "test_raw_frames"
            frame_dir = osp.join(self.data_prefix["img"], frame_dir, video_id)

            # keep the train videos
            if not self.test_mode and is_test:
                continue

            if not self.test_mode and not self.train_with_val and video_id in nexar_val:
                continue

            # keep the test videos
            if self.test_mode and video_id not in nexar_val and not is_test:
                continue

            data_list.append(
                dict(
                    frame_dir=frame_dir,
                    video_id=video_id,
                    target=target,
                    abnormal_start_frame=int(np.ceil(line[4] * fps)) if not is_test and target else None,
                    accident_frame=int(np.ceil(line[3] * fps)) if not is_test and target else None,
                    total_frames=int(line[2]),
                    fps=fps,
                    is_val=video_id in nexar_val,
                    is_test=is_test,
                )
            )
        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info["filename_tmpl"] = self.filename_tmpl
        return data_info


@DATASETS.register_module()
class MultiDataset(BaseDataset):
    def __init__(
        self,
        cap=None,
        dada=None,
        d2city=None,
        nexar=None,
        pipeline_video=None,
        pipeline_frame=None,
        test_mode=False,
        train_with_val=True,
        indices=None,
    ) -> None:
        self.cap = cap
        self.dada = dada
        self.d2city = d2city
        self.nexar = nexar

        self.test_mode = test_mode
        self.train_with_val = train_with_val
        self._indices = indices
        self._metainfo = self._load_metainfo(None)
        self.serialize_data = True
        self.max_refetch = 1000

        self.pipeline_video = Compose(pipeline_video)
        self.pipeline_frame = Compose(pipeline_frame)

        self.full_init()

    def load_data_list(self) -> List[dict]:
        data_list = []
        if self.cap:
            fin = pd.read_excel(osp.join(self.cap["data_root"], self.cap["ann_file"]), sheet_name=1).values.tolist()
            for line in fin:
                video_id = str(line[0]).zfill(6)
                1 - 10, 11, 12 - 42, 43, 44 - 62
                if 1 <= line[5] <= 10:
                    frame_dir = "1-10"
                elif line[5] == 11:
                    frame_dir = "11"
                elif 12 <= line[5] <= 42:
                    frame_dir = "12-42"
                elif line[5] == 43:
                    frame_dir = "43"
                elif 44 <= line[5] <= 62:
                    frame_dir = "44-62"
                frame_dir = osp.join(self.cap["data_root"], frame_dir, str(line[5]), video_id, "images")

                # skip the broken videos
                if video_id in ["011665", "008728", "004928", "007155"]:
                    continue

                # only for ego-car accidents
                if not 1 <= line[5] <= 18:
                    continue

                if line[5] not in [7, 8, 10, 11]:
                    continue

                # keep the train videos
                if self.test_mode:
                    continue

                correct_total_frames = {
                    "005722": 78,
                    "009073": 211,
                    "009074": 220,
                    "009480": 586,
                    "010087": 301,
                    "009541": 453,
                    "009222": 303,
                    "009255": 217,
                    "008589": 106,
                }

                # fix the total frames
                if video_id in correct_total_frames.keys():
                    line[10] = correct_total_frames[video_id]

                data_list.append(
                    dict(
                        dataset="cap",
                        filename=None,
                        frame_dir=frame_dir,
                        filename_tmpl=self.cap["filename_tmpl"],
                        start_index=1,
                        video_id=video_id,
                        target=True,
                        abnormal_start_frame=line[7],
                        accident_frame=line[9],
                        total_frames=line[10],
                        fps=get_fps(video_id, dataset_type="cap"),
                        is_val=False,
                        is_test=False,
                    )
                )

        if self.dada:
            fin = pd.read_excel(osp.join(self.dada["data_root"], self.dada["ann_file"]), sheet_name=1).values.tolist()
            for line in fin:
                video_id = str(line[0]).zfill(3)
                frame_dir = osp.join(self.dada["data_root"], str(line[5]), video_id, "images")
                video_id = f"{line[5]}_{video_id}"

                # skip the videos without accidents
                if line[8] == -1:
                    continue

                # only for ego-car accidents
                if not 1 <= line[5] <= 18:
                    continue

                if line[5] not in [7, 8, 10, 11]:
                    continue

                # keep the train videos
                if self.test_mode:
                    continue

                correct_total_frames = {
                    "5_040": 382,
                    "5_049": 204,
                    "6_009": 325,
                    "10_169": 320,
                    "11_076": 666,
                    "11_139": 220,
                    "36_002": 342,
                    "37_003": 695,
                    "43_080": 338,
                    "43_188": 220,
                    "48_065": 330,
                    "50_136": 422,
                    "56_008": 433,
                }

                # fix the total frames
                if video_id in correct_total_frames.keys():
                    line[10] = correct_total_frames[video_id]

                data_list.append(
                    dict(
                        dataset="dada",
                        filename=None,
                        frame_dir=frame_dir,
                        filename_tmpl=self.dada["filename_tmpl"],
                        start_index=1,
                        video_id=video_id,
                        target=True,
                        abnormal_start_frame=line[7],
                        accident_frame=line[8],
                        total_frames=line[10],
                        fps=get_fps(video_id, dataset_type="dada"),
                        is_val=False,
                        is_test=False,
                    )
                )

        if self.d2city:
            fin = pd.read_csv(osp.join(self.d2city["data_root"], self.d2city["ann_file"])).values.tolist()
            for line in fin:
                video_id = line[0]
                filename = osp.join(self.d2city["data_root"], "raw", str(int(line[1])).zfill(4), video_id + ".mp4")
                fps = 20 if np.random.rand() < 0.5 else 30

                # keep the train videos
                if self.test_mode:
                    continue

                if np.random.rand() < 0.8:
                    continue

                data_list.append(
                    dict(
                        dataset="d2city",
                        filename=filename,
                        frame_dir=None,
                        filename_tmpl=None,
                        start_index=0,
                        video_id=video_id,
                        target=False,
                        abnormal_start_frame=None,
                        accident_frame=None,
                        total_frames=int(line[2]),
                        fps=fps,
                        is_val=False,
                        is_test=False,
                    )
                )

        if self.nexar:
            fin = pd.read_csv(osp.join(self.nexar["data_root"], self.nexar["ann_file"])).values.tolist()
            for line in fin:
                video_id = str(int(line[0])).zfill(5)
                is_test = bool(line[1])
                target = bool(line[5]) if not np.isnan(line[5]) else None
                fps = 30
                if not is_test:
                    filename = "train"
                    frame_dir = "train_raw_frames"
                else:
                    filename = "test"
                    frame_dir = "test_raw_frames"
                filename = osp.join(self.nexar["data_root"], filename, video_id + ".mp4")
                frame_dir = osp.join(self.nexar["data_root"], frame_dir, video_id)

                # keep the train videos
                if not self.test_mode and is_test:
                    continue

                if not self.test_mode and not self.train_with_val and video_id in nexar_val:
                    continue

                # keep the test videos
                if self.test_mode and video_id not in nexar_val and not is_test:
                    continue

                data_list.append(
                    dict(
                        dataset="nexar",
                        filename=filename,
                        frame_dir=frame_dir,
                        filename_tmpl=self.nexar["filename_tmpl"],
                        start_index=0,
                        video_id=video_id,
                        target=target,
                        abnormal_start_frame=int(np.ceil(line[4] * fps)) if not is_test and target else None,
                        accident_frame=int(np.ceil(line[3] * fps)) if not is_test and target else None,
                        total_frames=int(line[2]),
                        fps=fps,
                        is_val=video_id in nexar_val,
                        is_test=is_test,
                    )
                )
        return data_list

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        if data_info["dataset"] in ["d2city", "nexar"]:
            return self.pipeline_video(data_info)
        else:
            return self.pipeline_frame(data_info)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info["modality"] = "RGB"
        return data_info
