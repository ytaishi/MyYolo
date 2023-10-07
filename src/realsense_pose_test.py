import pyrealsense2 as rs
import numpy as np
import cv2
import torch  # このライブラリは今回の例では不要ですが、他の用途で使うかもしれないので残しています。
import json
import serial
import os

import math


class RealSenseManager:
    def __init__(self, config):
        self.pipeline = rs.pipeline()
        self.config = config
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.gyro)
        self.config.enable_stream(rs.stream.accel)

    def start_streaming(self):
        self.pipeline.start(self.config)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return depth_frame, color_frame

    # 姿勢情報の取得
    def get_pose(self):
        frames = self.pipeline.wait_for_frames()
        gyro_frame = frames.first(rs.stream.gyro)
        accel_frame = frames.first(rs.stream.accel)

        # ジャイロのデータを取得
        gyro_data = gyro_frame.as_motion_frame().get_motion_data()
        # 加速度のデータを取得
        accel_data = accel_frame.as_motion_frame().get_motion_data()

        return gyro_data, accel_data

    def stop_streaming(self):
        self.pipeline.stop()


class PoseEstimator:
    def __init__(self):
        # 直上から見た時の4つの角の座標を定義する (例: 1280x720の画像の場合)
        self.dst_points = np.array(
            [[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float32
        )

    def get_pitch_roll(self, accel_data):
        ax, ay, az = accel_data.x, accel_data.y, accel_data.z

        pitch = math.atan2(ay, math.sqrt(ax * ax + az * az))
        roll = math.atan2(-ax, az)

        # ラジアンから度に変換
        pitch_degrees = math.degrees(pitch)
        roll_degrees = math.degrees(roll)

        return pitch_degrees, roll_degrees

    def estimate_corners(self, pitch, roll, image_width, image_height):
        # 姿勢情報から、変換後の画像の4つの角の座標を推定する (簡略化のため疑似コード)
        # 実際には、この部分での詳細な計算が必要です。
        # 以下は、ダミーの座標を返す例です。
        return np.array(
            [[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]],
            dtype=np.float32,
        )

    def perspective_transform(self, image, src_points):
        # 射影変換行列を計算する
        M = cv2.getPerspectiveTransform(src_points, self.dst_points)
        transformed_image = cv2.warpPerspective(
            image, M, (image.shape[1], image.shape[0])
        )
        return transformed_image


if __name__ == "__main__":
    # RealSenseの設定
    config = rs.config()

    # RealSenseマネージャを初期化
    manager = RealSenseManager(config)
    pose_estimator = PoseEstimator()

    manager.start_streaming()

    try:
        while True:
            # フレームを取得
            depth_frame, color_frame = manager.get_frames()

            # カラー画像を取得
            color_image = np.asanyarray(color_frame.get_data())

            # 姿勢情報を取得
            gyro_data, accel_data = manager.get_pose()
            # print(f"ジャイロ: {gyro_data}, 加速度: {accel_data}")

            pitch, roll = pose_estimator.get_pitch_roll(accel_data)
            print(f"pitch: {pitch}, roll: {roll}")

            src_points = pose_estimator.estimate_corners(
                pitch, roll, color_image.shape[1], color_image.shape[0]
            )

            transformed_image = pose_estimator.perspective_transform(
                color_image, src_points
            )

            cv2.imshow("Transformed RealSense", transformed_image)

            # キー入力で終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        manager.stop_streaming()
        cv2.destroyAllWindows()
