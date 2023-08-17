import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import json
import serial
import os


# RealSenseカメラの管理とデータストリームの処理
class RealSenseManager:
    def __init__(self, config):
        self.pipeline = rs.pipeline()
        self.config = config
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # ストリームの開始
    def start_streaming(self):
        self.pipeline.start(self.config)

    # フレームの取得
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return depth_frame, color_frame

    # ストリームの停止
    def stop_streaming(self):
        self.pipeline.stop()


# 物体検出と距離の計算
class ObjectDetector:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.rect_threshold = int(1280 / 8 * 720 / 8)
        self.range_value = 3

    # 距離の計算
    def calculate_distance(self, depth_image, x, y):
        x, y = int(x), int(y)
        region = depth_image[
            y - self.range_value : y + self.range_value + 1,
            x - self.range_value : x + self.range_value + 1,
        ]
        distances = region[region != 0]
        return np.mean(distances) if distances.size != 0 else 0

    # 物体の検出
    def detect_objects(self, img, depth_image):
        result = self.model(img)
        result.render()
        detected_objects = []
        for detection in result.xyxy[0]:
            if (detection[2] - detection[0]) * (
                detection[3] - detection[1]
            ) < self.rect_threshold:
                continue
            label = result.names[int(detection[5])]
            x_center = (detection[0] + detection[2]) / 2
            y_center = (detection[1] + detection[3]) / 2
            distance = self.calculate_distance(depth_image, x_center, y_center)
            detected_objects.append((label, distance))
        return detected_objects


# 検出データの整理と送信
class DataOrganizer:
    def __init__(self, output_dir, use_serial_communication=False):
        self.output_dir = output_dir
        self.use_serial_communication = use_serial_communication
        if self.use_serial_communication:
            self.ser = serial.Serial("COM4", 115200)

    # 物体と距離の情報を辞書形式で整形する関数
    def organize_data(self, detected_objects):
        detected_data = {}
        for _, obj in enumerate(sorted(detected_objects, key=lambda x: x[1])[:3]):
            detected_data[obj[0]] = round(obj[1] / 1000, 2) if obj[1] != 0 else "NA"

        # 3つ未満の場合はNAで埋める
        while len(detected_data) < 3:
            detected_data[f"NA_{len(detected_data)}"] = "NA"

        return detected_data

    # データの送信
    def send_data(self, detected_data):
        data_to_send = json.dumps(detected_data)
        print(data_to_send)
        if self.use_serial_communication:
            self.ser.write((data_to_send + "\r").encode("utf-8"))
        with open(f"{self.output_dir}/output.json", "a") as file:
            file.write(data_to_send + "\r")


# 画像の表示
class DisplayManager:
    def __init__(self):
        pass

    # 画像の表示
    def display(self, color_image, depth_image):
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", images)
        cv2.waitKey(1)


# メイン関数
def main():
    USE_SERIAL_COMMUNICATION = False
    OUTPUT_DIR = "./out"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = rs.config()
    realsense_manager = RealSenseManager(config)
    object_detector = ObjectDetector()
    data_organizer = DataOrganizer(OUTPUT_DIR, USE_SERIAL_COMMUNICATION)

    display_manager = DisplayManager()

    realsense_manager.start_streaming()
    try:
        while True:
            depth_frame, color_frame = realsense_manager.get_frames()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            hole_filling = rs.hole_filling_filter(1)
            depth_frame = hole_filling.process(depth_frame)

            detected_objects = object_detector.detect_objects(color_image, depth_image)
            detected_data = data_organizer.organize_data(detected_objects)
            data_organizer.send_data(detected_data)

            display_manager.display(color_image, depth_image)
    finally:
        realsense_manager.stop_streaming()


if __name__ == "__main__":
    main()
