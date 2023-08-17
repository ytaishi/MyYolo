# https://murasan-net.com/index.php/2022/10/22/realsence-d435-pytorch-predict/

import pyrealsense2 as rs
import numpy as np
import cv2
import torch

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# YOLOのモデルをロード
model = torch.hub.load("ultralytics/yolov5", "yolov5s")


# PyTorchを使った物体検出
def predict(img):
    # 推論を実行
    result = model(img)
    result.render()

    # 戻り値
    return result.ims[0]


# メイン処理
def main():
    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            # print(depth_frame.get_distance(int(640 / 2), int(480 / 2)))
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())

            # 推論実行
            color_image = predict(color_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", images)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
