#!/usr/bin/env python3
"""
replay_plot.py — Re-draws the roofline chart from the last benchmark run
without re-running any benchmarks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from exp_drawer import plot_rooflines

cpu_peak = 30.7
cpu_bw   = 16.5

npu_measured_modes = [
    {"name": "base",    "peak_gops": 15951, "bw_gbs": 3.9, "color": "#d62728"},
    {"name": "global4", "peak_gops":  4990, "bw_gbs": 0.7, "color": "#ff7f0e"},
    {"name": "global8", "peak_gops":  8227, "bw_gbs": 0.7, "color": "#9467bd"},
]

all_npu_points = [
    # base
    {"base_name": "MobileNet_V2",        "mode_name": "base",    "perf":   173, "ai":   68.4, "color": "#d62728"},
    {"base_name": "ResNet50",            "mode_name": "base",    "perf":  1551, "ai":  158.1, "color": "#d62728"},
    {"base_name": "DenseNet121",         "mode_name": "base",    "perf":   752, "ai":  208.0, "color": "#d62728"},
    {"base_name": "VGG16",              "mode_name": "base",    "perf":  3922, "ai":  223.5, "color": "#d62728"},
    {"base_name": "ConvNeXt_Tiny",       "mode_name": "base",    "perf":  1026, "ai":  303.3, "color": "#d62728"},
    {"base_name": "ViT_Base_Patch16_224","mode_name": "base",    "perf":   691, "ai":  191.2, "color": "#d62728"},
    {"base_name": "YOLO11s",             "mode_name": "base",    "perf":  2133, "ai": 2320.4, "color": "#d62728"},
    {"base_name": "YOLO11l",             "mode_name": "base",    "perf":  3980, "ai": 3509.8, "color": "#d62728"},
    {"base_name": "YOLO11sSeg",          "mode_name": "base",    "perf":  2805, "ai": 3213.4, "color": "#d62728"},
    {"base_name": "YOLO11lPose",         "mode_name": "base",    "perf":  4001, "ai": 3317.2, "color": "#d62728"},
    # global4
    {"base_name": "ResNet50",            "mode_name": "global4", "perf":  1855, "ai":  157.8, "color": "#ff7f0e"},
    {"base_name": "DenseNet121",         "mode_name": "global4", "perf":  1336, "ai":  205.6, "color": "#ff7f0e"},
    {"base_name": "VGG16",              "mode_name": "global4", "perf":  5043, "ai":  223.4, "color": "#ff7f0e"},
    {"base_name": "ConvNeXt_Tiny",       "mode_name": "global4", "perf":  1810, "ai":  300.1, "color": "#ff7f0e"},
    {"base_name": "ViT_Base_Patch16_224","mode_name": "global4", "perf":   934, "ai":  190.1, "color": "#ff7f0e"},
    {"base_name": "YOLO11s",             "mode_name": "global4", "perf":  4377, "ai": 2268.0, "color": "#ff7f0e"},
    {"base_name": "YOLO11l",             "mode_name": "global4", "perf":  8218, "ai": 3448.4, "color": "#ff7f0e"},
    {"base_name": "YOLO11sSeg",          "mode_name": "global4", "perf":  4106, "ai": 3143.2, "color": "#ff7f0e"},
    {"base_name": "YOLO11lPose",         "mode_name": "global4", "perf":  8295, "ai": 3262.5, "color": "#ff7f0e"},
    # global8
    {"base_name": "ResNet50",            "mode_name": "global8", "perf":  2016, "ai":  157.6, "color": "#9467bd"},
    {"base_name": "DenseNet121",         "mode_name": "global8", "perf":  1318, "ai":  204.2, "color": "#9467bd"},
    {"base_name": "VGG16",              "mode_name": "global8", "perf":  5478, "ai":  223.4, "color": "#9467bd"},
    {"base_name": "ConvNeXt_Tiny",       "mode_name": "global8", "perf":  2030, "ai":  298.2, "color": "#9467bd"},
    {"base_name": "ViT_Base_Patch16_224","mode_name": "global8", "perf":  1411, "ai":  189.3, "color": "#9467bd"},
    {"base_name": "YOLO11s",             "mode_name": "global8", "perf":  4138, "ai": 2229.7, "color": "#9467bd"},
    {"base_name": "YOLO11l",             "mode_name": "global8", "perf": 10158, "ai": 3401.9, "color": "#9467bd"},
    {"base_name": "YOLO11sSeg",          "mode_name": "global8", "perf":  5729, "ai": 3089.7, "color": "#9467bd"},
    {"base_name": "YOLO11lPose",         "mode_name": "global8", "perf": 10359, "ai": 3221.4, "color": "#9467bd"},
    # multi (batch=4 corrected)
    {"base_name": "MobileNet_V2",        "mode_name": "multi",   "perf":   585, "ai":   68.2, "color": "#8c564b"},
    {"base_name": "ResNet50",            "mode_name": "multi",   "perf":  3253, "ai":  157.0, "color": "#8c564b"},
    {"base_name": "DenseNet121",         "mode_name": "multi",   "perf":  2223, "ai":  201.9, "color": "#8c564b"},
    {"base_name": "VGG16",              "mode_name": "multi",   "perf":  8682, "ai":  223.3, "color": "#8c564b"},
    {"base_name": "ConvNeXt_Tiny",       "mode_name": "multi",   "perf":  3019, "ai":  295.9, "color": "#8c564b"},
    {"base_name": "ViT_Base_Patch16_224","mode_name": "multi",   "perf":  1894, "ai":  188.4, "color": "#8c564b"},
    {"base_name": "YOLO11s",             "mode_name": "multi",   "perf":  5031, "ai": 2233.6, "color": "#8c564b"},
    {"base_name": "YOLO11l",             "mode_name": "multi",   "perf": 10491, "ai": 3411.7, "color": "#8c564b"},
    {"base_name": "YOLO11sSeg",          "mode_name": "multi",   "perf":  5982, "ai": 3098.6, "color": "#8c564b"},
    {"base_name": "YOLO11lPose",         "mode_name": "multi",   "perf": 10695, "ai": 3230.2, "color": "#8c564b"},
]

plot_rooflines(cpu_peak, cpu_bw, all_npu_points, cpu_models=[],
               npu_measured_modes=npu_measured_modes,
               filename="roofline_experimental.png")
