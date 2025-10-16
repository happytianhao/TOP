import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from sklearn.metrics import average_precision_score


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    计算两个边界框的IoU (Intersection over Union)
    
    参数:
    - bbox1, bbox2: 形状为 (4,) 的数组，格式为 [cx, cy, w, h]
    
    返回:
    - IoU值，范围 [0, 1]
    """
    # 将中心点格式转换为左上角格式
    x1_1, y1_1 = bbox1[0] - bbox1[2] / 2, bbox1[1] - bbox1[3] / 2
    x2_1, y2_1 = bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2
    
    x1_2, y1_2 = bbox2[0] - bbox2[2] / 2, bbox2[1] - bbox2[3] / 2
    x2_2, y2_2 = bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def build_results(
    reference_csv_path: str,
    results_csv_path: str,
    with_bbox: bool = False,
) -> Dict[str, object]:
    """
    从参考CSV与结果CSV构建评估所需结果字典。

    参数:
    - reference_csv_path: 参考CSV文件路径
    - results_csv_path: 结果CSV文件路径  
    - with_bbox: 是否包含bbox信息，如果为True则处理bbox相关字段

    逻辑说明：
    - 先读取结果CSV，建立 (video_id, frame_id) -> score 的查找索引。
    - 全量遍历参考CSV，按 (video_id, have_accident) 聚合为同一clip，仅收集帧序列。
    - 先校验所有clip的帧数是否一致；若不一致抛出ValueError并给出帧数集合。
    - 通过校验后，再逐clip逐帧到索引中检索score；若某帧在结果中缺失，抛出KeyError。
    - 如果with_bbox=True，还会处理bbox相关字段并计算IoU指标。

    返回：
    - results["video_id"]: List[str]，每个clip一个条目（相同video_id可能出现多次）。
    - results["frame_id"]: List[List[int]]，与video_id按索引对齐的帧列表。
    - results["have_accident"]: numpy.ndarray[int]，每clip一个值。
    - results["score"]: numpy.ndarray[float]，二维数组，形状=(clip数, 每clip帧数)。
    - 如果with_bbox=True，还会包含：
      - results["target_bbox"]: numpy.ndarray[float]，形状=(clip数, 每clip帧数, 4)，目标bbox
      - results["pred_bbox"]: numpy.ndarray[float]，形状=(clip数, 每clip帧数, 4)，预测bbox
      - results["have_target"]: numpy.ndarray[int]，形状=(clip数, 每clip帧数)，是否有目标
      - results["have_pred"]: numpy.ndarray[int]，形状=(clip数, 每clip帧数)，是否有预测
      - results["valid_pred"]: numpy.ndarray[int]，形状=(clip数, 每clip帧数)，预测是否有效
    """
    # 建立 (video_id, frame_id) -> score 的索引
    key_to_score: Dict[Tuple[str, int], float] = {}
    key_to_bbox: Dict[Tuple[str, int], np.ndarray] = {}  # 用于存储bbox信息
    
    with open(results_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("结果文件为空: {}".format(results_csv_path))
        
        # 根据with_bbox参数与新旧格式确定期望的最小列数
        # 新格式：无bbox=4列(video_id, frame_id, have_accident, score)；有bbox=8列(+ cx, cy, w, h)
        # 旧格式：无bbox=3列(video_id, frame_id, score)；有bbox=7列(score在第3列)
        if with_bbox:
            min_expected_cols = 7  # 兼容旧格式最小为7
        else:
            min_expected_cols = 3  # 兼容旧格式最小为3
            
        for row in reader:
            if not row:
                continue
            if len(row) < min_expected_cols:
                continue
                
            vid = row[0]
            frame = int(row[1])
            # 解析score与bbox（兼容新旧格式）
            if with_bbox:
                # 新格式>=8列: have_accident在row[2]，score在row[3]，bbox从row[4:8]
                # 旧格式==7列: score在row[2]，bbox从row[3:7]
                if len(row) >= 8:
                    score = float(row[3])
                    bbox_cols = row[4:8]
                else:
                    score = float(row[2])
                    bbox_cols = row[3:7]
            else:
                # 新格式>=4列: have_accident在row[2]，score在row[3]
                # 旧格式==3列: score在row[2]
                if len(row) >= 4:
                    score = float(row[3])
                else:
                    score = float(row[2])
            key_to_score[(vid, frame)] = score
            
            # 如果包含bbox信息，解析bbox（兼容新旧格式）
            if with_bbox:
                if len(row) >= 8:
                    cx, cy, w, h = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                    key_to_bbox[(vid, frame)] = np.array([cx, cy, w, h])
                elif len(row) >= 7:
                    cx, cy, w, h = float(row[3]), float(row[4]), float(row[5]), float(row[6])
                    key_to_bbox[(vid, frame)] = np.array([cx, cy, w, h])

    # 全量扫描参考文件：按 (video_id, have_accident) 聚合同一clip，仅收集帧（不查score）
    grouped_by_clip: Dict[Tuple[str, int], Dict[str, List]] = {}
    key_to_target_bbox: Dict[Tuple[str, int], np.ndarray] = {}  # 用于存储目标bbox信息
    
    with open(reference_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("参考文件为空: {}".format(reference_csv_path))
        
        # 根据with_bbox参数确定期望的表头
        if with_bbox:
            # 期望表头: video_id,frame_id,have_accident,cx,cy,w,h
            expected_cols = 7
        else:
            # 期望表头: video_id,frame_id,have_accident
            expected_cols = 3
            
        for row in reader:
            if not row:
                continue
            if len(row) < expected_cols:
                continue
                
            vid = row[0]
            frame = int(row[1])
            have = int(row[2])
            key_clip = (vid, have)
            if key_clip not in grouped_by_clip:
                grouped_by_clip[key_clip] = {"frames": []}
            grouped_by_clip[key_clip]["frames"].append(frame)
            
            # 如果包含bbox信息，解析目标bbox
            if with_bbox and len(row) >= 7:
                cx, cy, w, h = float(row[3]), float(row[4]), float(row[5]), float(row[6])
                key_to_target_bbox[(vid, frame)] = np.array([cx, cy, w, h])

    # 校验所有clip的帧数是否一致（例如都应为固定长度，如20）
    clip_lengths = [len(v["frames"]) for v in grouped_by_clip.values()]
    if clip_lengths:
        unique_lengths = set(clip_lengths)
        if len(unique_lengths) > 1:
            raise ValueError(
                "存在帧数不一致的clip: 发现的帧数集合={}".format(sorted(unique_lengths))
            )

    # 组织输出（保持插入顺序尽量按照首次出现顺序；字典在Py3.7+保持插入序）
    video_ids: List[str] = []
    frame_groups: List[List[int]] = []
    score_groups: List[np.ndarray] = []
    have_list: List[int] = []
    
    # bbox相关的输出列表
    target_bbox_groups: List[np.ndarray] = []
    pred_bbox_groups: List[np.ndarray] = []
    have_target_groups: List[np.ndarray] = []
    have_pred_groups: List[np.ndarray] = []
    valid_pred_groups: List[np.ndarray] = []

    # 帧数校验通过后，再逐clip逐帧检索score并组织输出
    for (vid, have) in grouped_by_clip.keys():
        data = grouped_by_clip[(vid, have)]
        frames = list(data["frames"])  # 按出现顺序
        # 通过索引在校验通过后再检索score，若缺失则抛错
        scores_for_clip = []
        
        # bbox相关的数组
        target_bbox_for_clip = []
        pred_bbox_for_clip = []
        have_target_for_clip = []
        have_pred_for_clip = []
        valid_pred_for_clip = []
        
        for frame in frames:
            key = (vid, frame)
            if key not in key_to_score:
                raise KeyError(
                    "未在结果文件中找到匹配行: video_id={}, frame_id={}".format(vid, frame)
                )
            scores_for_clip.append(key_to_score[key])
            
            # 处理bbox相关信息
            if with_bbox:
                # 处理目标bbox
                if key in key_to_target_bbox:
                    target_bbox = key_to_target_bbox[key]
                    # 检查是否有有效目标（所有值都>=0）
                    have_target = 1 if np.all(target_bbox >= 0) else 0
                    target_bbox_for_clip.append(target_bbox)
                else:
                    target_bbox = np.array([-1.0, -1.0, -1.0, -1.0])
                    have_target = 0
                    target_bbox_for_clip.append(target_bbox)
                have_target_for_clip.append(have_target)
                
                # 处理预测bbox
                if key in key_to_bbox:
                    pred_bbox = key_to_bbox[key]
                    # 检查是否有有效预测（所有值都>=0）
                    have_pred = 1 if np.all(pred_bbox >= 0) else 0
                    pred_bbox_for_clip.append(pred_bbox)
                else:
                    pred_bbox = np.array([-1.0, -1.0, -1.0, -1.0])
                    have_pred = 0
                    pred_bbox_for_clip.append(pred_bbox)
                have_pred_for_clip.append(have_pred)
                
                # 计算valid_pred
                if have_target == 1 and have_pred == 1:
                    iou = calculate_iou(target_bbox, pred_bbox)
                    valid_pred = 1 if iou > 0.5 else 0
                else:
                    valid_pred = 0
                valid_pred_for_clip.append(valid_pred)

        video_ids.append(vid)
        frame_groups.append(frames)
        score_groups.append(np.array(scores_for_clip, dtype=float))
        have_list.append(int(have))
        
        # 添加bbox相关的数据
        if with_bbox:
            target_bbox_groups.append(np.array(target_bbox_for_clip, dtype=float))
            pred_bbox_groups.append(np.array(pred_bbox_for_clip, dtype=float))
            have_target_groups.append(np.array(have_target_for_clip, dtype=int))
            have_pred_groups.append(np.array(have_pred_for_clip, dtype=int))
            valid_pred_groups.append(np.array(valid_pred_for_clip, dtype=int))

    results = {
        "video_id": video_ids,
        "frame_id": frame_groups,
        "have_accident": np.array(have_list, dtype=int),
        "score": np.stack(score_groups, axis=0) if score_groups else np.zeros((0, 0), dtype=float),
    }
    
    # 如果包含bbox信息，添加bbox相关的字段
    if with_bbox:
        results["target_bbox"] = np.stack(target_bbox_groups, axis=0) if target_bbox_groups else np.zeros((0, 0, 4), dtype=float)
        results["pred_bbox"] = np.stack(pred_bbox_groups, axis=0) if pred_bbox_groups else np.zeros((0, 0, 4), dtype=float)
        results["have_target"] = np.stack(have_target_groups, axis=0) if have_target_groups else np.zeros((0, 0), dtype=int)
        results["have_pred"] = np.stack(have_pred_groups, axis=0) if have_pred_groups else np.zeros((0, 0), dtype=int)
        results["valid_pred"] = np.stack(valid_pred_groups, axis=0) if valid_pred_groups else np.zeros((0, 0), dtype=int)
    
    return results


def area_under_curve(y_true: np.ndarray, y_scores: np.ndarray, fpr_max: float = 1.0):
    # 1. 按预测得分降序排序，并记录真实标签
    sorted_indices = np.argsort(y_scores)[::-1]  # 从高到低排序
    y_true_sorted = y_true[sorted_indices]

    # 2. 计算正负样本数量
    P = np.sum(y_true == 1)  # 正样本数
    N = np.sum(y_true == 0)  # 负样本数

    # 3. 初始化TPR和FPR
    TPR = [0]  # 真正例率（初始为0）
    FPR = [0]  # 假正例率（初始为0）
    TP, FP = 0, 0  # 累积真正例和假正例数

    # 4. 遍历排序后的样本，动态更新TPR和FPR
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            TP += 1  # 真正例+1
        else:
            FP += 1  # 假正例+1
        TPR.append(TP / P if P > 0 else 0.0)  # 计算当前TPR
        FPR.append(FP / N if N > 0 else 0.0)  # 计算当前FPR

    # 5. 梯形法计算AUC（积分ROC曲线下面积）
    auc_value = 0.0
    for i in range(1, len(FPR)):
        if FPR[i] > fpr_max:
            break
        dx = FPR[i] - FPR[i - 1]  # x轴宽度
        dy = TPR[i] + TPR[i - 1]  # y轴平均高度
        auc_value += dx * dy / 2  # 梯形面积累加

    return auc_value / fpr_max if fpr_max > 0 else 0.0, FPR, TPR


def compute_auc_ap(results: Dict[str, object], with_bbox: bool = False) -> Tuple[Dict[str, float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    基于二维scores与have_accident，计算以下分数组合与指标：
    - 分数组合：
      score_neg(负样本全帧max)、score_pos(正样本全帧max)、score_n(负样本倒数1-5帧max)、
      score_0(正样本倒数1-5帧max)、score_5(正样本倒数6-10帧max)、score_10(正样本倒数11-15帧max)、score_15(正样本倒数16-20帧max)
    - 指标：
      AUC（score_neg vs score_pos）、
      AUC_0.0s/AUC_0.5s/AUC_1.0s/AUC_1.5s（均为score_n vs 对应正样本窗口）、mAUC（0.5/1.0/1.5均值，fpr_max=1）
      AUC^0.1系列（同上但fpr_max=0.1，含mAUC^0.1）
      AP系列（同上配对，含mAP=0.5/1.0/1.5均值）

    参数:
    - results: 包含评估数据的字典
    - with_bbox: 是否使用bbox信息；True时正样本仅统计valid_pred=1帧

    返回：
    - 评估指标字典
    - ROC曲线数据字典，包含各score组合的(FPR, TPR)数据
    """
    scores: np.ndarray = results["score"]  # (C, T)
    has_acc: np.ndarray = results["have_accident"]  # (C,)

    if scores.ndim != 2:
        raise ValueError("results['score'] 必须是二维数组，当前 ndim={}".format(scores.ndim))

    pos_mask = has_acc == 1
    neg_mask = has_acc == 0

    # 如果使用bbox信息，获取valid_pred
    if with_bbox:
        if "valid_pred" not in results:
            raise ValueError("with_bbox=True 但 results 中缺少 'valid_pred' 字段")
        valid_pred: np.ndarray = results["valid_pred"]  # (C, T)
        if valid_pred.shape != scores.shape:
            raise ValueError("valid_pred 的形状 {} 与 scores 的形状 {} 不匹配".format(valid_pred.shape, scores.shape))

        # 正样本仅统计valid_pred=1的帧
        pos_scores_masked = scores[pos_mask].copy()
        pos_valid_mask = valid_pred[pos_mask]
        pos_scores_masked[pos_valid_mask == 0] = 0

        score_neg = np.max(scores[neg_mask], axis=1) if np.any(neg_mask) else np.zeros((0,), dtype=float)
        score_pos = np.max(pos_scores_masked, axis=1) if np.any(pos_mask) else np.zeros((0,), dtype=float)
    else:
        score_neg = np.max(scores[neg_mask], axis=1) if np.any(neg_mask) else np.zeros((0,), dtype=float)
        score_pos = np.max(scores[pos_mask], axis=1) if np.any(pos_mask) else np.zeros((0,), dtype=float)

    # 通用窗口函数：从末尾偏移 start_from_end，窗口大小 window
    def window_max(sample_mask: np.ndarray, start_from_end: int, window: int = 5) -> np.ndarray:
        if not np.any(sample_mask):
            return np.zeros((0,), dtype=float)
        T = scores.shape[1]
        if T < start_from_end + window:
            raise ValueError("clip帧数={} 小于所需窗口(end-{} ~ end-{}]".format(T, start_from_end + window, start_from_end))
        s = T - (start_from_end + window)
        e = T - start_from_end
        # 正样本在 with_bbox 模式下使用已mask的分数
        if with_bbox and np.array_equal(sample_mask, pos_mask):
            base = pos_scores_masked
        else:
            base = scores[sample_mask]
        return np.max(base[:, s:e], axis=1)

    # 依据定义计算窗口分数
    score_n = window_max(neg_mask, 0, 5)                 # 负样本倒数1-5帧
    score_0 = window_max(pos_mask, 0, 5)                 # 正样本倒数1-5帧
    score_5 = window_max(pos_mask, 5, 5)                 # 正样本倒数6-10帧
    score_10 = window_max(pos_mask, 10, 5)               # 正样本倒数11-15帧
    score_15 = window_max(pos_mask, 15, 5)               # 正样本倒数16-20帧

    def auc_with(neg: np.ndarray, pos: np.ndarray, fpr_max: float = 1.0) -> Tuple[float, np.ndarray, np.ndarray]:
        if neg.size == 0 or pos.size == 0:
            return float('nan'), np.array([]), np.array([])
        y_scores = np.concatenate([neg, pos], axis=0)
        y_true = np.concatenate([np.zeros_like(neg, dtype=int), np.ones_like(pos, dtype=int)], axis=0)
        auc_value, fpr, tpr = area_under_curve(y_true=y_true, y_scores=y_scores, fpr_max=fpr_max)
        return float(auc_value), fpr, tpr

    def ap_with(neg: np.ndarray, pos: np.ndarray) -> float:
        if neg.size == 0 or pos.size == 0:
            return float('nan')
        y_scores = np.concatenate([neg, pos], axis=0)
        y_true = np.concatenate([np.zeros_like(neg, dtype=int), np.ones_like(pos, dtype=int)], axis=0)
        return float(average_precision_score(y_true, y_scores))

    # 计算 fpr_max=1.0 的指标和ROC曲线数据
    auc, fpr, tpr = auc_with(score_neg, score_pos)
    auc_0, fpr_0, tpr_0 = auc_with(score_n, score_0)
    auc_5, fpr_5, tpr_5 = auc_with(score_n, score_5)
    auc_10, fpr_10, tpr_10 = auc_with(score_n, score_10)
    auc_15, fpr_15, tpr_15 = auc_with(score_n, score_15)
    valid_auc = [x for x in [auc_5, auc_10, auc_15] if not np.isnan(x)]
    mauc = float(np.mean(valid_auc)) if valid_auc else float('nan')

    # 计算 fpr_max=0.1 的指标
    auc_01, _, _ = auc_with(score_neg, score_pos, fpr_max=0.1)
    auc_01_0, _, _ = auc_with(score_n, score_0, fpr_max=0.1)
    auc_01_5, _, _ = auc_with(score_n, score_5, fpr_max=0.1)
    auc_01_10, _, _ = auc_with(score_n, score_10, fpr_max=0.1)
    auc_01_15, _, _ = auc_with(score_n, score_15, fpr_max=0.1)
    valid_auc_01 = [x for x in [auc_01_5, auc_01_10, auc_01_15] if not np.isnan(x)]
    mauc_01 = float(np.mean(valid_auc_01)) if valid_auc_01 else float('nan')

    ap = ap_with(score_neg, score_pos)
    ap_0 = ap_with(score_n, score_0)
    ap_5 = ap_with(score_n, score_5)
    ap_10 = ap_with(score_n, score_10)
    ap_15 = ap_with(score_n, score_15)
    valid_ap = [x for x in [ap_5, ap_10, ap_15] if not np.isnan(x)]
    map = float(np.mean(valid_ap)) if valid_ap else float('nan')

    metrics = {
        "AUC": auc,
        "AUC_0.0s": auc_0,
        "AUC_0.5s": auc_5,
        "AUC_1.0s": auc_10,
        "AUC_1.5s": auc_15,
        "mAUC": mauc,
        "AUC^0.1": auc_01,
        "AUC^0.1_0.0s": auc_01_0,
        "AUC^0.1_0.5s": auc_01_5,
        "AUC^0.1_1.0s": auc_01_10,
        "AUC^0.1_1.5s": auc_01_15,
        "mAUC^0.1": mauc_01,
        "AP": ap,
        "AP_0.0s": ap_0,
        "AP_0.5s": ap_5,
        "AP_1.0s": ap_10,
        "AP_1.5s": ap_15,
        "mAP": map,
    }

    # 组织ROC曲线数据
    rocs = {
        "roc": (fpr, tpr),
        "roc_0": (fpr_0, tpr_0),
        "roc_5": (fpr_5, tpr_5),
        "roc_10": (fpr_10, tpr_10),
        "roc_15": (fpr_15, tpr_15),
    }

    return metrics, rocs


def compute_tta(results: Dict[str, object], fps: float = 10.0, with_bbox: bool = False) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    计算TTA：
    - 使用所有负样本clip的所有帧分数拼接并按降序作为阈值序列 thresholds。
    - 仅对正样本clip计算TTA：
      对每个阈值，计算正样本clip中第一个 >= 阈值 的帧索引 idx；
      若不存在则该正样本clip的TTA记为0；否则 TTA_clip = (T - idx) / fps。
      对该阈值下所有正样本clip的TTA取均值；
    - 最终对所有阈值的均值再取平均，得到总TTA。
    
    参数:
    - results: 包含评估数据的字典
    - fps: 帧率，用于计算时间
    - with_bbox: 是否使用bbox信息，如果为True，则对于正样本只考虑valid_pred=1的帧
    
    返回：
    - ttas: 一个字典 {"TTA": 不截断TTA, "TTA^0.1": 截断到FAR<=0.1 的TTA}
    - tta_means_per_threshold数组（每个阈值对应的平均TTA）
    - far_means_per_threshold数组（每个阈值对应的平均FAR值）
    """
    scores: np.ndarray = results["score"]  # (C, T)
    has_acc: np.ndarray = results["have_accident"]  # (C,)

    if scores.ndim != 2:
        raise ValueError("results['score'] 必须是二维数组")
    C, T = scores.shape
    if C == 0 or T == 0:
        return {"TTA": float('nan'), "TTA^0.1": float('nan')}, np.array([]), np.array([])

    neg_mask = (has_acc == 0)
    pos_mask = (has_acc == 1)
    if not np.any(neg_mask) or not np.any(pos_mask):
        return {"TTA": float('nan'), "TTA^0.1": float('nan')}, np.array([]), np.array([])

    # 阈值序列：所有负样本帧分数降序
    thresholds = np.sort(scores[neg_mask].ravel())[::-1]
    if thresholds.size == 0:
        return {"TTA": float('nan'), "TTA^0.1": float('nan')}, np.array([]), np.array([])
    if thresholds.size == 0:
        return {"TTA": float('nan'), "TTA^0.1": float('nan')}, np.array([]), np.array([])

    # 向量化计算每个阈值下的TTA：
    # meet[c, k, t] 表示第 c 个正样本clip在阈值 k 时第 t 帧是否 >= 阈值
    pos_scores = scores[pos_mask]  # (C_pos, T)
    C_pos = pos_scores.shape[0]
    
    # 如果使用bbox信息，对正样本分数进行mask
    if with_bbox:
        if "valid_pred" not in results:
            raise ValueError("with_bbox=True 但 results 中缺少 'valid_pred' 字段")
        valid_pred: np.ndarray = results["valid_pred"]  # (C, T)
        if valid_pred.shape != scores.shape:
            raise ValueError("valid_pred 的形状 {} 与 scores 的形状 {} 不匹配".format(valid_pred.shape, scores.shape))
        
        # 对正样本分数进行mask，只考虑valid_pred=1的帧
        pos_scores_masked = pos_scores.copy()  # 复制正样本分数
        pos_valid_mask = valid_pred[pos_mask]  # 正样本的valid_pred
        pos_scores_masked[pos_valid_mask == 0] = 0  # 将valid_pred=0的帧分数设为0
        
        meet = pos_scores_masked[:, None, :] >= thresholds[None, :, None]  # (C_pos, K, T)
    else:
        meet = pos_scores[:, None, :] >= thresholds[None, :, None]  # (C_pos, K, T)
    
    has_meet = meet.any(axis=2)  # (C_pos, K)
    first_idx = np.argmax(meet, axis=2)  # (C_pos, K)

    # 计算每个阈值、每个正样本clip的 TTA，并在无满足时置0
    tta_per_clip = np.zeros_like(first_idx, dtype=float)  # (C_pos, K)
    tta_per_clip[has_meet] = (T - first_idx[has_meet]) / fps

    # 先在clip维度求均值，再对阈值求均值
    tta_means_per_threshold = tta_per_clip.mean(axis=0)  # (K,)
    
    # 计算FAR值：对于每个阈值，计算负样本中分数>=阈值的比例
    neg_scores_flat = scores[neg_mask].ravel()  # 所有负样本分数
    far_means_per_threshold = np.array([np.mean(neg_scores_flat >= threshold) for threshold in thresholds])
    
    # 不截断聚合
    tta_full = float(tta_means_per_threshold.mean()) if tta_means_per_threshold.size > 0 else float('nan')

    # 截断聚合（FAR<=0.1）
    if tta_means_per_threshold.size > 0 and far_means_per_threshold.size > 0:
        valid_mask = far_means_per_threshold <= 0.1
        if np.any(valid_mask):
            tta_fpr = float(tta_means_per_threshold[valid_mask].mean())
        else:
            tta_fpr = float('nan')
    else:
        tta_fpr = float('nan')
    return {"TTA": tta_full, "TTA^0.1": tta_fpr}, tta_means_per_threshold, far_means_per_threshold


def plot_tta_vs_far(tta_means_per_threshold: np.ndarray, far_means_per_threshold: np.ndarray, output_path: str = None):
    """
    绘制TTA vs FAR图表
    
    参数:
    - tta_means_per_threshold: 每个阈值对应的平均TTA数组
    - far_means_per_threshold: 每个阈值对应的平均FAR数组  
    - output_path: 保存路径，如果为None则显示图表
    
    图表说明:
    - 横坐标: False Alarm Rate (FAR), 范围0-1
    - 纵坐标: Time to Accident (TTA)
    - 蓝色线条: TTA vs FAR曲线
    """
    # 创建图表
    plt.figure(figsize=(10, 8))
    plt.plot(far_means_per_threshold, tta_means_per_threshold, 'b-', linewidth=2, label='TTA vs FAR')
    
    # 设置坐标轴
    plt.xlabel('False Alarm Rate (FAR)', fontsize=12)
    plt.ylabel('Time to Accident (TTA)', fontsize=12)
    plt.title('TTA vs FAR Curve', fontsize=14, fontweight='bold')
    
    # 设置坐标轴范围
    plt.xlim(0, 1)
    plt.ylim(0, 2)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend()
    
    # 保存或显示图表
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(rocs: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: str = None):
    """
    绘制ROC曲线图表
    
    参数:
    - rocs: ROC曲线数据字典，包含各score组合的(FPR, TPR)数据
    - output_path: 保存路径，如果为None则显示图表
    
    图表说明:
    - 横坐标: False Alarm Rate (FAR), 范围0-1
    - 纵坐标: True Positive Rate (TPR/Recall), 范围0-1
    - 不同颜色的线条: 不同时间提前量的事故预测ROC曲线
    """
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 定义颜色和标签
    colors = ['blue', 'purple', 'red', 'green', 'orange']
    labels = {
        'roc': 'Anticipation (Ahead Accident)',
        'roc_0': 'Anticipation (0.0s Ahead Accident)',
        'roc_5': 'Anticipation (0.5s Ahead Accident)',
        'roc_10': 'Anticipation (1.0s Ahead Accident)',
        'roc_15': 'Anticipation (1.5s Ahead Accident)'
    }
    
    # 绘制每条ROC曲线
    for i, (key, (fpr, tpr)) in enumerate(rocs.items()):
        if len(fpr) > 0 and len(tpr) > 0:
            plt.plot(fpr, tpr, color=colors[i], linewidth=2, label=labels.get(key, key))
    
    # 绘制对角线（随机分类器）
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    
    # 设置坐标轴
    plt.xlabel('False Alarm Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR/Recall)', fontsize=12)
    plt.title('ROC Curves for Accident Anticipation at Different Time Horizons', fontsize=14, fontweight='bold')
    
    # 设置坐标轴范围
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend()
    
    # 保存或显示图表
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_results_clips(
    results: Dict[str, object],
    video_ids: List[str],
    have_accidents: List[int],
    output_dir: str,
    data_root: str = "data/MM-AU/CAP-DATA",
    ann_file: str = "cap_val_annotations.csv",
    filename_tmpl: str = "{:06}.jpg",
    fps: float = 10.0,
    with_bbox: bool = False,
    threshold: float = 0.5,
    frame_size: Tuple[int, int] = (640, 360)
) -> None:
    """
    可视化results中的指定clips，兼容有无bbox的版本
    
    参数:
    - results: 包含评估数据的字典
    - video_ids: 要可视化的视频ID列表
    - have_accidents: 对应的have_accident值列表
    - output_dir: 输出视频的目录
    - data_root: 数据根目录，默认为"data/MM-AU/CAP-DATA"
    - ann_file: 注释文件名，默认为"cap_val_annotations.csv"
    - filename_tmpl: 帧文件名模板，默认为"{:06}.jpg"
    - fps: 生成视频的帧率（与注释文件中的原视频帧率不同）
    - with_bbox: 是否包含bbox信息
    - threshold: 分数阈值，用于高亮显示
    - frame_size: 输出视频的帧大小 (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证输入参数
    if len(video_ids) != len(have_accidents):
        raise ValueError("video_ids 和 have_accidents 的长度必须相同")
    
    # 加载注释文件信息
    ann_file_path = os.path.join(data_root, ann_file)
    video_info = {}  # video_id -> {path, fps, total_frames, abnormal_start_frame, accident_occur_frame, have_accident}
    
    if os.path.exists(ann_file_path):
        with open(ann_file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError("注释文件为空: {}".format(ann_file_path))
            
            for row in reader:
                if not row or len(row) < 8:
                    continue
                
                vid = row[0]
                path = row[1]
                original_fps = float(row[3])  # 原视频帧率
                total_frames = int(row[4])
                abnormal_start_frame = int(row[5])
                accident_occur_frame = int(row[6])
                have_accident = int(row[7])
                
                # 使用(video_id, have_accident)作为键，因为同一个video_id可能有多个clip
                key = (vid, have_accident)
                video_info[key] = {
                    'path': path,
                    'original_fps': original_fps,  # 重命名为original_fps避免冲突
                    'total_frames': total_frames,
                    'abnormal_start_frame': abnormal_start_frame,
                    'accident_occur_frame': accident_occur_frame,
                    'have_accident': have_accident
                }
    else:
        print(f"警告: 注释文件不存在: {ann_file_path}")
    
    # 获取数据
    all_video_ids = results["video_id"]
    all_have_accidents = results["have_accident"]
    scores = results["score"]  # (C, T)
    
    # 获取bbox相关信息（如果可用）
    target_bbox = None
    pred_bbox = None
    have_target = None
    have_pred = None
    valid_pred = None
    
    if with_bbox:
        if "target_bbox" in results:
            target_bbox = results["target_bbox"]  # (C, T, 4)
        if "pred_bbox" in results:
            pred_bbox = results["pred_bbox"]  # (C, T, 4)
        if "have_target" in results:
            have_target = results["have_target"]  # (C, T)
        if "have_pred" in results:
            have_pred = results["have_pred"]  # (C, T)
        if "valid_pred" in results:
            valid_pred = results["valid_pred"]  # (C, T)
    
    # 为每个指定的clip创建视频
    for i, (video_id, have_accident) in enumerate(zip(video_ids, have_accidents)):
        # 查找对应的clip索引
        clip_idx = None
        for j, (vid, acc) in enumerate(zip(all_video_ids, all_have_accidents)):
            if vid == video_id and acc == have_accident:
                clip_idx = j
                break
        
        if clip_idx is None:
            print(f"警告: 未找到视频 {video_id} 且 have_accident={have_accident} 的clip")
            continue
        
        # 获取该clip的数据
        clip_scores = scores[clip_idx]  # (T,)
        clip_frames = results["frame_id"][clip_idx]  # List[int]
        
        # 获取bbox信息（如果可用）
        clip_target_bbox = target_bbox[clip_idx] if target_bbox is not None else None  # (T, 4)
        clip_pred_bbox = pred_bbox[clip_idx] if pred_bbox is not None else None  # (T, 4)
        clip_have_target = have_target[clip_idx] if have_target is not None else None  # (T,)
        clip_have_pred = have_pred[clip_idx] if have_pred is not None else None  # (T,)
        clip_valid_pred = valid_pred[clip_idx] if valid_pred is not None else None  # (T,)
        
        # 创建视频文件路径
        bbox_suffix = "_bbox" if with_bbox else ""
        video_path = os.path.join(output_dir, f"clip_{video_id}_{have_accident}{bbox_suffix}.mp4")
        
        # 创建VideoWriter - 使用函数参数中的fps（生成视频的帧率）
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        
        print(f"正在处理视频: {video_id}, have_accident={have_accident}")
        
        # 获取当前clip的视频信息
        clip_key = (video_id, have_accident)
        current_video_info = video_info.get(clip_key, {})
        
        # 处理每一帧
        for frame_idx, (frame_id, score) in enumerate(zip(clip_frames, clip_scores)):
            # 创建帧图像
            # 优先使用注释文件中的path信息构建帧路径
            if 'path' in current_video_info:
                # 使用注释文件中的path: data_root/path/filename_tmpl.format(frame_id)
                frame_path = os.path.join(data_root, current_video_info['path'], filename_tmpl.format(frame_id))
            else:
                # 回退到原来的方式: data_root/video_id/filename_tmpl.format(frame_id)
                frame_path = os.path.join(data_root, video_id, filename_tmpl.format(frame_id))
            
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                frame = cv2.resize(frame, frame_size)
            else:
                # 如果帧文件不存在，创建黑色背景
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            
            # 绘制状态条带 - 使用注释文件中的准确时间信息
            if have_accident == 1 and 'abnormal_start_frame' in current_video_info and 'accident_occur_frame' in current_video_info:
                # 事故视频：使用注释文件中的准确帧信息
                abnormal_start = current_video_info['abnormal_start_frame']
                accident_occur = current_video_info['accident_occur_frame']
                
                if frame_id < abnormal_start:
                    # 异常开始前为安全
                    cv2.rectangle(frame, (0, 0), (frame_size[0], 36), (0, 255, 0), -1)
                    status_text = "Safe Scenario"
                elif frame_id < accident_occur:
                    # 异常开始到事故发生之间
                    cv2.rectangle(frame, (0, 0), (frame_size[0], 36), (255, 0, 0), -1)
                    status_text = "Anomaly Appeared"
                else:
                    # 事故发生及之后
                    cv2.rectangle(frame, (0, 0), (frame_size[0], 36), (0, 0, 255), -1)
                    status_text = "Accident Occurred"
            else:
                # 安全视频或没有注释信息的事故视频，统一显示为安全
                cv2.rectangle(frame, (0, 0), (frame_size[0], 36), (0, 255, 0), -1)
                status_text = "Safe Scenario"
            
            # 添加状态文本
            cv2.putText(frame, status_text, (frame_size[0] - 220, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            
            # 绘制分数信息
            score_text = f"Score: {score:.3f}"
            cv2.putText(frame, score_text, (20, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            
            # 如果分数超过阈值，绘制红色边框
            if score >= threshold:
                cv2.rectangle(frame, (0, 0), (frame_size[0], frame_size[1]), (0, 0, 255), 5)
            
            # 绘制分数条形图
            bar_width = int(score * 200)  # 最大宽度200像素
            cv2.rectangle(frame, (20, 50), (20 + bar_width, 70), (0, 255, 255), -1)
            cv2.rectangle(frame, (20, 50), (220, 70), (255, 255, 255), 2)
            
            # 如果包含bbox信息，绘制bbox矩形
            if with_bbox and clip_target_bbox is not None:
                def draw_bbox_center_xywh(img, bbox_xywh, color, thickness=2):
                    cx, cy, w, h = float(bbox_xywh[0]), float(bbox_xywh[1]), float(bbox_xywh[2]), float(bbox_xywh[3])
                    # 兼容归一化坐标与像素坐标：若 w,h 均 <= 1，则视为归一化到 [0,1]
                    if 0 < w <= 1.0 and 0 < h <= 1.0:
                        px_w = w * frame_size[0]
                        px_h = h * frame_size[1]
                        px_cx = cx * frame_size[0]
                        px_cy = cy * frame_size[1]
                    else:
                        px_w = w
                        px_h = h
                        px_cx = cx
                        px_cy = cy
                    x1 = int(px_cx - px_w / 2)
                    y1 = int(px_cy - px_h / 2)
                    x2 = int(px_cx + px_w / 2)
                    y2 = int(px_cy + px_h / 2)
                    # 边界裁剪
                    x1 = max(0, min(frame_size[0] - 1, x1))
                    y1 = max(0, min(frame_size[1] - 1, y1))
                    x2 = max(0, min(frame_size[0] - 1, x2))
                    y2 = max(0, min(frame_size[1] - 1, y2))
                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                # 目标框（绿色）
                if clip_have_target is not None and clip_have_target[frame_idx] == 1:
                    target_bbox_info = clip_target_bbox[frame_idx]
                    draw_bbox_center_xywh(frame, target_bbox_info, color=(0, 255, 0), thickness=2)

                # 预测框（蓝色，若无效可用红色描边）
                if clip_have_pred is not None and clip_have_pred[frame_idx] == 1:
                    pred_bbox_info = clip_pred_bbox[frame_idx]
                    # 若提供了有效性标记，用颜色区分
                    if clip_valid_pred is not None and clip_valid_pred[frame_idx] == 0:
                        pred_color = (0, 0, 255)  # 无效：红色
                    else:
                        pred_color = (255, 0, 0)  # 有效：蓝色
                    draw_bbox_center_xywh(frame, pred_bbox_info, color=pred_color, thickness=2)
            
            # 添加帧信息和注释文件中的时间信息
            frame_info = f"Frame: {frame_id} ({frame_idx+1}/{len(clip_frames)})"
            cv2.putText(frame, frame_info, (20, frame_size[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 如果有注释信息，显示关键帧信息和原视频帧率
            if 'abnormal_start_frame' in current_video_info and 'accident_occur_frame' in current_video_info:
                abnormal_start = current_video_info['abnormal_start_frame']
                accident_occur = current_video_info['accident_occur_frame']
                original_fps = current_video_info.get('original_fps', 10.0)  # 原视频帧率
                
                timing_info = f"Abnormal: {abnormal_start}, Accident: {accident_occur}"
                cv2.putText(frame, timing_info, (20, frame_size[1] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 显示原视频帧率信息
                fps_info = f"Original FPS: {original_fps:.1f}, Output FPS: {fps:.1f}"
                cv2.putText(frame, fps_info, (20, frame_size[1] - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # 显示距离关键帧的帧数和时间（基于原视频fps）
                if frame_id < abnormal_start:
                    frames_to_abnormal = abnormal_start - frame_id
                    time_to_abnormal = frames_to_abnormal / original_fps
                    countdown_info = f"Frames to abnormal: {frames_to_abnormal} ({time_to_abnormal:.1f}s)"
                    cv2.putText(frame, countdown_info, (20, frame_size[1] - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                elif frame_id < accident_occur:
                    frames_to_accident = accident_occur - frame_id
                    time_to_accident = frames_to_accident / original_fps
                    countdown_info = f"Frames to accident: {frames_to_accident} ({time_to_accident:.1f}s)"
                    cv2.putText(frame, countdown_info, (20, frame_size[1] - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    frames_since_accident = frame_id - accident_occur
                    time_since_accident = frames_since_accident / original_fps
                    countdown_info = f"Frames since accident: {frames_since_accident} ({time_since_accident:.1f}s)"
                    cv2.putText(frame, countdown_info, (20, frame_size[1] - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # 写入帧到视频
            video_writer.write(frame)
        
        # 释放VideoWriter
        video_writer.release()
        print(f"视频已保存到: {video_path}")


if __name__ == "__main__":
    # 测试不带bbox的版本
    reference_csv = \
        "/home/zhaotianhao/Code/TOP/data/MM-AU/CAP-DATA/cap_val_references.csv"
    results_csv = \
        "/home/zhaotianhao/Code/TOP/cap_val_results.csv"

    print("=== 测试不带bbox的版本 ===")
    results = build_results(reference_csv, results_csv)
    # 简要打印验证
    print("clips:", len(results["video_id"]))
    print("have_accident shape:", results["have_accident"].shape)
    print("scores shape:", results["score"].shape)
    if results["video_id"]:
        print("first clip video_id:", results["video_id"][0])
        print("first clip frames:", len(results["frame_id"][0]))
        print("first clip scores shape:", results["score"][0].shape)
    
    # 测试带bbox的版本
    print("\n=== 测试带bbox的版本 ===")
    reference_bbox_csv = \
        "/home/zhaotianhao/Code/TOP/data/MM-AU/CAP-DATA/cap_val_references_bbox.csv"
    results_bbox_csv = \
        "/home/zhaotianhao/Code/TOP/cap_val_results_bbox.csv"
    
    results_bbox = build_results(reference_bbox_csv, results_bbox_csv, with_bbox=True)
    print("clips:", len(results_bbox["video_id"]))
    print("have_accident shape:", results_bbox["have_accident"].shape)
    print("scores shape:", results_bbox["score"].shape)
    print("target_bbox shape:", results_bbox["target_bbox"].shape)
    print("pred_bbox shape:", results_bbox["pred_bbox"].shape)
    print("have_target shape:", results_bbox["have_target"].shape)
    print("have_pred shape:", results_bbox["have_pred"].shape)
    print("valid_pred shape:", results_bbox["valid_pred"].shape)
    
    if results_bbox["video_id"]:
        print("first clip video_id:", results_bbox["video_id"][0])
        print("first clip frames:", len(results_bbox["frame_id"][0]))
        print("first clip have_target:", results_bbox["have_target"][0])
        print("first clip have_pred:", results_bbox["have_pred"][0])
        print("first clip valid_pred:", results_bbox["valid_pred"][0])

    # 计算并打印AUC与AP
    metrics, rocs = compute_auc_ap(results)
    print("Metrics (without bbox):", metrics)
    
    # 计算并打印带bbox的AUC与AP
    if results_bbox:
        metrics_bbox, rocs_bbox = compute_auc_ap(results_bbox, with_bbox=True)
        print("Metrics (with bbox):", metrics_bbox)
    # 计算并打印TTA
    ttas, tta_means_per_threshold, far_means_per_threshold = compute_tta(results)
    print("TTA (without bbox):", ttas.get("TTA", float('nan')))
    print("TTA^0.1 (without bbox):", ttas.get("TTA^0.1", float('nan')))
    print(f"TTA数组长度: {len(tta_means_per_threshold)}")
    print(f"FAR数组长度: {len(far_means_per_threshold)}")
    print(f"FAR范围: {far_means_per_threshold.min():.4f} - {far_means_per_threshold.max():.4f}")
    print(f"TTA范围: {tta_means_per_threshold.min():.4f} - {tta_means_per_threshold.max():.4f}")
    
    # 计算并打印带bbox的TTA
    if results_bbox:
        ttas_bbox, tta_means_per_threshold_bbox, far_means_per_threshold_bbox = compute_tta(results_bbox, with_bbox=True)
        print("TTA (with bbox):", ttas_bbox.get("TTA", float('nan')))
        print("TTA^0.1 (with bbox):", ttas_bbox.get("TTA^0.1", float('nan')))
        if not np.isnan(ttas_bbox.get("TTA", float('nan'))) and not np.isnan(ttas.get("TTA", float('nan'))):
            print(f"TTA差异 (bbox - 无bbox): {ttas_bbox['TTA'] - ttas['TTA']:+.4f}")
    
    # 绘制TTA vs FAR图表并保存
    output_dir = os.path.dirname(results_csv)
    tta_output_path = os.path.join(output_dir, 'tta_vs_far.png')
    plot_tta_vs_far(tta_means_per_threshold, far_means_per_threshold, tta_output_path)
    
    # 绘制ROC曲线图表并保存
    roc_output_path = os.path.join(output_dir, 'roc_curves.png')
    plot_roc_curves(rocs, roc_output_path)
