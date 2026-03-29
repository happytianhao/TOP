# Copyright (c) OpenMMLab. All rights reserved.
import os
import matplotlib.pyplot as plt
from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class EpochHook(Hook):
    def before_train_epoch(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        if hasattr(model, 'epoch'):
            model.epoch = runner.epoch
        if hasattr(model, 'cls_head') and hasattr(model.cls_head, 'epoch'):
            model.cls_head.epoch = runner.epoch

    def before_val_epoch(self, runner) -> None:
        for metric in runner.val_evaluator.metrics:
            metric.epoch = runner.epoch
            metric.log_dir = runner.log_dir


@HOOKS.register_module()
class AnticipationMetricHook(Hook):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.fpr_train = []
        self.tta_train = []
        self.mtta_train = []
        self.AUC0_train = []
        self.AUC5_train = []
        self.AUC10_train = []
        self.AUC15_train = []
        self.mAUC_train = []
        self.fpr_val = []
        self.tta_val = []
        self.mtta_val = []
        self.AUC0_val = []
        self.AUC5_val = []
        self.AUC10_val = []
        self.AUC15_val = []
        self.mAUC_val = []

    def after_val_epoch(self, runner, metrics) -> None:
        self.epochs.append(runner.epoch)
        plt.figure()
        if "\nfpr#0.5" in metrics and "tta#0.5" in metrics and "mtta#0.1" in metrics:
            self.fpr_train.append(metrics["\nfpr#0.5"])
            self.tta_train.append(metrics["tta#0.5"])
            self.mtta_train.append(metrics["mtta#0.1"])
            plt.plot(self.epochs, self.fpr_train, label="fpr#0.5 (train)", marker="+", color="red")
            plt.plot(self.epochs, self.tta_train, label="tta#0.5 (train)", marker="+", color="blue")
            plt.plot(self.epochs, self.mtta_train, label="mtta#0.1 (train)", marker="+", color="green")
        if "\nfpr@0.5" in metrics and "tta@0.5" in metrics and "mtta@0.1" in metrics:
            self.fpr_val.append(metrics["\nfpr@0.5"])
            self.tta_val.append(metrics["tta@0.5"])
            self.mtta_val.append(metrics["mtta@0.1"])
            plt.plot(self.epochs, self.fpr_val, label="fpr@0.5 (val)", marker="o", color="red")
            plt.plot(self.epochs, self.tta_val, label="tta@0.5 (val)", marker="o", color="blue")
            plt.plot(self.epochs, self.mtta_val, label="mtta@0.1 (val)", marker="o", color="green")
        plt.title("Anticipation Metrics")
        plt.xlabel("Epochs")
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(os.path.join(runner.log_dir, "metrics_tta.png"))
        plt.close()

        plt.figure()
        if "mAUC#" in metrics:
            self.AUC0_train.append(metrics["AUC#0.0s"])
            self.AUC5_train.append(metrics["AUC#0.5s"])
            self.AUC10_train.append(metrics["AUC#1.0s"])
            self.AUC15_train.append(metrics["AUC#1.5s"])
            self.mAUC_train.append(metrics["mAUC#"])
            plt.plot(self.epochs, self.AUC0_train, label="AUC#0.0s (train)", marker="+", color="purple")
            plt.plot(self.epochs, self.AUC5_train, label="AUC#0.5s (train)", marker="+", color="blue")
            plt.plot(self.epochs, self.AUC10_train, label="AUC#1.0s (train)", marker="+", color="red")
            plt.plot(self.epochs, self.AUC15_train, label="AUC#1.5s (train)", marker="+", color="green")
            plt.plot(self.epochs, self.mAUC_train, label="mAUC# (train)", marker="+", color="orange")
        if "mAUC@" in metrics:
            self.AUC0_val.append(metrics["AUC@0.0s"])
            self.AUC5_val.append(metrics["AUC@0.5s"])
            self.AUC10_val.append(metrics["AUC@1.0s"])
            self.AUC15_val.append(metrics["AUC@1.5s"])
            self.mAUC_val.append(metrics["mAUC@"])
            plt.plot(self.epochs, self.AUC0_val, label="AUC@0.0s (val)", marker="o", color="purple")
            plt.plot(self.epochs, self.AUC5_val, label="AUC@0.5s (val)", marker="o", color="blue")
            plt.plot(self.epochs, self.AUC10_val, label="AUC@1.0s (val)", marker="o", color="red")
            plt.plot(self.epochs, self.AUC15_val, label="AUC@1.5s (val)", marker="o", color="green")
            plt.plot(self.epochs, self.mAUC_val, label="mAUC@ (val)", marker="o", color="orange")
        i_v = self.mAUC_val.index(max(self.mAUC_val))
        plt.title(f"mAUC_val@{i_v+1}={self.mAUC_val[i_v]:.4f}")
        plt.xlabel("Epochs")
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(os.path.join(runner.log_dir, "metrics_AUC.png"))
        plt.close()


@HOOKS.register_module()
class UnifiedMetricHook(Hook):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.auc_series_full = None
        self.auc_series_01 = None
        self.ap_series = None
        self.tta_series = None
        self.tta01_series = None

    def after_val_epoch(self, runner, metrics) -> None:
        self.epochs.append(runner.epoch)
        trends_dir = os.path.join(runner.log_dir, "trends")
        os.makedirs(trends_dir, exist_ok=True)

        # AUC (fpr_max=1.0)
        auc_keys = [
            ("AUC", "AUC (full)"),
            ("AUC_0.0s", "AUC_0.0s (full)"),
            ("AUC_0.5s", "AUC_0.5s (full)"),
            ("AUC_1.0s", "AUC_1.0s (full)"),
            ("AUC_1.5s", "AUC_1.5s (full)"),
            ("mAUC", "mAUC (full)"),
        ]
        if any(k in metrics for k, _ in auc_keys):
            if self.auc_series_full is None:
                self.auc_series_full = {k: [] for k, _ in auc_keys}
            for k, _ in auc_keys:
                self.auc_series_full[k].append(metrics.get(k, float("nan")))
            plt.figure()
            for k, label in auc_keys:
                plt.plot(self.epochs, self.auc_series_full[k], label=label, marker="o")
            plt.title("AUC Trends (fpr_max=1.0)")
            plt.xlabel("Epochs")
            plt.ylabel("AUC")
            plt.ylim(0, 1)
            plt.yticks([i * 0.1 for i in range(0, 11)])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(trends_dir, "AUCs.png"))
            plt.close()

        # AUC^0.1 (fpr_max=0.1)
        auc01_keys = [
            ("AUC^0.1", "AUC (0.1)"),
            ("AUC^0.1_0.0s", "AUC_0.0s (0.1)"),
            ("AUC^0.1_0.5s", "AUC_0.5s (0.1)"),
            ("AUC^0.1_1.0s", "AUC_1.0s (0.1)"),
            ("AUC^0.1_1.5s", "AUC_1.5s (0.1)"),
            ("mAUC^0.1", "mAUC (0.1)"),
        ]
        if any(k in metrics for k, _ in auc01_keys):
            if self.auc_series_01 is None:
                self.auc_series_01 = {k: [] for k, _ in auc01_keys}
            for k, _ in auc01_keys:
                self.auc_series_01[k].append(metrics.get(k, float("nan")))
            plt.figure()
            for k, label in auc01_keys:
                plt.plot(self.epochs, self.auc_series_01[k], label=label, marker="o")
            plt.title("AUC^0.1 Trends (fpr_max=0.1)")
            plt.xlabel("Epochs")
            plt.ylabel("AUC^0.1")
            plt.ylim(0, 1)
            plt.yticks([i * 0.1 for i in range(0, 11)])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(trends_dir, "AUC^0.1s.png"))
            plt.close()

        # AP
        ap_keys = [
            ("AP", "AP"),
            ("AP_0.0s", "AP_0.0s"),
            ("AP_0.5s", "AP_0.5s"),
            ("AP_1.0s", "AP_1.0s"),
            ("AP_1.5s", "AP_1.5s"),
            ("mAP", "mAP"),
        ]
        if any(k in metrics for k, _ in ap_keys):
            if self.ap_series is None:
                self.ap_series = {k: [] for k, _ in ap_keys}
            for k, _ in ap_keys:
                self.ap_series[k].append(metrics.get(k, float("nan")))
            plt.figure()
            for k, label in ap_keys:
                plt.plot(self.epochs, self.ap_series[k], label=label, marker="o")
            plt.title("AP Trends")
            plt.xlabel("Epochs")
            plt.ylabel("AP")
            plt.ylim(0, 1)
            plt.yticks([i * 0.1 for i in range(0, 11)])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(trends_dir, "APs.png"))
            plt.close()

        # TTA 与 TTA^0.1
        if "TTA" in metrics or "TTA^0.1" in metrics:
            if self.tta_series is None:
                self.tta_series = []
            if self.tta01_series is None:
                self.tta01_series = []
            self.tta_series.append(metrics.get("TTA", float("nan")))
            self.tta01_series.append(metrics.get("TTA^0.1", float("nan")))
            plt.figure()
            plt.plot(self.epochs, self.tta_series, label="TTA", marker="o")
            plt.plot(self.epochs, self.tta01_series, label="TTA^0.1", marker="o")
            plt.title("TTA Trends")
            plt.xlabel("Epochs")
            plt.ylabel("TTA")
            plt.ylim(0, 2)
            plt.yticks([i * 0.2 for i in range(0, 11)])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(trends_dir, "TTAs.png"))
            plt.close()
