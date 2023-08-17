import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time

# ストリームの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# YOLOのモデルをロード
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# 矩形の最小サイズのしきい値（ピクセル単位）
RECT_THRESHOLD = int(640 / 8 * 480 / 8)


# 距離の平均を計算する関数
def calculate_distance(depth_image, x, y):
    distances = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            distance = depth_image[int(y) + i, int(x) + j]
            if distance != 0:
                distances.append(distance)
    return sum(distances) / len(distances) if distances else 0


# PyTorchを使った物体検出
def predict(img, depth_image):
    # 推論を実行
    result = model(img)
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
        distance = calculate_distance(depth_image, x_center, y_center)  # 距離の計算
        detected_objects.append((label, distance))

    # 距離が近い順にソートし、0mのものは除外
    detected_objects = [
        obj for obj in sorted(detected_objects, key=lambda x: x[1]) if obj[1] != 0
    ]

    # 3つまで表示、小数点2桁まで
    for i, (label, distance) in enumerate(detected_objects[:3]):
        print(f"{i + 1}: {label}: {round(distance / 1000, 2)}m")

    return result.ims[0]


def main():
    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 画像をnumpy配列に変換
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

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

            # 更新周期を1Hzに設定
            # time.sleep(1)

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
