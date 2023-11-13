#!/usr/bin/env python
#! coding:utf-8

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time
import json
import serial
import os


# シリアル通信するかどうかのフラグ
USE_SERIAL_COMMUNICATION = False

# Bluetoothシリアル設定
if USE_SERIAL_COMMUNICATION:
    ser = serial.Serial("COM4", 115200)

# 出力ファイルのディレクトリ作成
OUTPUT_DIR = "./out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ストリームの設定
PIPELINE = rs.pipeline()
CONFIG = rs.config()
CONFIG.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
CONFIG.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# YOLOのモデルをロード
MODEL = torch.hub.load("ultralytics/yolov5", "yolov5s")

# 矩形の最小サイズのしきい値（ピクセル単位）
RECT_THRESHOLD = int(1280 / 8 * 720 / 8)

# 平均化範囲
RANGE_VALUE = 3


# 距離の平均を計算する関数
def calculate_distance(depth_image, x, y, range_value):
    x, y = int(x), int(y)
    region = depth_image[
        y - range_value : y + range_value + 1, x - range_value : x + range_value + 1
    ]
    distances = region[region != 0]
    return np.mean(distances) if distances.size != 0 else 0


# 物体と距離の情報を辞書形式で整形する関数
def organize_detected_data(detected_objects):
    detected_data = {}
    for _, obj in enumerate(sorted(detected_objects, key=lambda x: x[1])[:3]):
        detected_data[obj[0]] = round(obj[1] / 1000, 2) if obj[1] != 0 else "NA"

    # 3つ未満の場合はNAで埋める
    while len(detected_data) < 3:
        detected_data[f"NA_{len(detected_data)}"] = "NA"

    return detected_data


# PyTorchを使った物体検出
def predict(img, depth_image):
    # 推論を実行
    result = MODEL(img)
    result.render()

    # 物体と距離の情報を取得
    detected_objects = []
    for detection in result.xyxy[0]:
        # 矩形のサイズがしきい値以下の場合は無視
        if (detection[2] - detection[0]) * (
            detection[3] - detection[1]
        ) < RECT_THRESHOLD:
            continue
        label = result.names[int(detection[5])]  # ラベル名のインデックスは5
        x_center = (detection[0] + detection[2]) / 2
        y_center = (detection[1] + detection[3]) / 2
        distance = calculate_distance(
            depth_image, x_center, y_center, RANGE_VALUE
        )  # 距離の計算
        detected_objects.append((label, distance))

    detected_data = organize_detected_data(detected_objects)

    # JSON形式でデータを作成
    data_to_send = json.dumps(detected_data)

    # Bluetooth経由でデータを送信
    if USE_SERIAL_COMMUNICATION:
        ser.write((data_to_send + "\r").encode("utf-8"))

    # ファイルに結果を書き込み
    with open(f"{OUTPUT_DIR}/output.json", "a") as file:
        file.write(data_to_send)

    # 画面にも表示
    for key, value in detected_data.items():
        print(f"{key}: {value}m" if value != "NA" else f"{key}: {value}")

    return result.ims[0]


def main():
    # ストリームの開始
    PIPELINE.start(CONFIG)

    try:
        while True:
            # 一対のフレーム（深度とカラー）を待つ
            frames = PIPELINE.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 画像をnumpy配列に変換
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 穴埋め処理
            hole_filling = rs.hole_filling_filter(1)
            depth_frame = hole_filling.process(depth_frame)

            # 推論実行
            color_image = predict(color_image, depth_image)

            # 深度画像をカラーマップで表示
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # 並べて表示
            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", images)
            cv2.waitKey(1)

            # time.sleep(1)

    finally:
        # ストリーミングの停止
        PIPELINE.stop()


if __name__ == "__main__":
    main()
