import cv2
import numpy as np
import math
from pylsd.lsd import lsd


# 射影変換を行う関数
def perspective_transform(image):
    src = np.float32([[400, 0], [880, 0], [0, 720], [1280, 720]])
    dst = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (1280, 720))


# 白色を検出する関数
def detect_white(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower_white, upper_white)


# 前処理 (ぼかし) 関数
def preprocess_mask(mask):
    return cv2.GaussianBlur(mask, (17, 17), 0)


# 線分検出 (LSD) 関数
def detect_lines(mask):
    return lsd(mask)  # 修正: PyLSDを使用


# 2つの点の距離を計算する関数
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 2つの線分がマージ可能か判断する関数
def can_merge(line1, line2, angle_thresh=10, distance_thresh=20):
    x1, y1, x2, y2 = line1[:4]
    angle1 = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

    x3, y3, x4, y4 = line2[:4]
    angle2 = math.atan2(y4 - y3, x4 - x3) * 180 / math.pi

    if abs(angle1 - angle2) > angle_thresh:
        return False

    # 線分間の各端点間の距離を計算
    d1 = distance(x1, y1, x3, y3)
    d2 = distance(x1, y1, x4, y4)
    d3 = distance(x2, y2, x3, y3)
    d4 = distance(x2, y2, x4, y4)

    if min(d1, d2, d3, d4) > distance_thresh:
        return False

    return True


# 線分をマージする関数
def merge_lines(lines, angle_thresh=10, distance_thresh=20):
    merged = []

    for line1 in lines:
        merged_flag = False
        for idx, line2 in enumerate(merged):
            if can_merge(line1, line2, angle_thresh, distance_thresh):
                x1, y1, x2, y2 = line1[:4]
                x3, y3, x4, y4 = line2[:4]
                # 端点の平均を取って新しい線分を作る
                new_line = np.array(
                    [(x1 + x3) / 2, (y1 + y3) / 2, (x2 + x4) / 2, (y2 + y4) / 2]
                )
                merged[idx] = new_line
                merged_flag = True
                break

        if not merged_flag:
            merged.append(np.array(line1[:4]))  # Numpy配列に変換してから追加

    return np.array(merged)  # 全要素が同じ形状であることを確認してからNumpy配列に変換


# 長さによって線分をフィルタリングする関数
def filter_long_lines(lines, length_thresh=50):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[:4]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 線分の長さを計算
        if length > length_thresh:  # 長さが閾値より大きい場合に追加
            filtered_lines.append(line)
    return np.array(filtered_lines)


# 平行性と白線の条件をチェックする関数
def filter_parallel_white_lines(
    lines, mask, angle_thresh=10, distance_thresh=20, white_thresh=0.8
):
    filtered_lines = []
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i >= j:
                continue
            x1, y1, x2, y2 = map(int, line1[:4])
            x3, y3, x4, y4 = map(int, line2[:4])

            # 角度を計算
            angle1 = math.atan2(y2 - y1, x2 - x1) * (180.0 / np.pi)
            angle2 = math.atan2(y4 - y3, x4 - x3) * (180.0 / np.pi)

            if abs(angle1 - angle2) < angle_thresh:
                # 平行な線を見つけたら、距離を計算
                distance = np.linalg.norm(
                    np.cross([x2 - x1, y2 - y1], [x1 - x3, y1 - y3])
                ) / np.linalg.norm([x2 - x1, y2 - y1])
                if distance < distance_thresh:
                    # 平行かつ近い線を見つけたら、その内部が白いかをチェック
                    x_mid = int((x1 + x3) / 2)
                    y_mid = int((y1 + y3) / 2)
                    if mask[y_mid, x_mid] > white_thresh * 255:
                        filtered_lines.append(line1)
                        filtered_lines.append(line2)
    return np.array(filtered_lines)


# 線分を描画する関数
def draw_lines(image, lines):
    if lines is not None:
        for line in lines:  # 修正: すでに2次元配列なのでそのままループ
            x0, y0, x1, y1 = map(int, line[:4])  # 修正: 最初の4つの要素だけを使用
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        transformed_image = perspective_transform(frame)  # 射影変換
        mask = detect_white(transformed_image)  # 白色を検出
        mask = preprocess_mask(mask)  # 前処理 (ぼかし)
        detected_lines = detect_lines(mask)  # 線分検出 (LSD) 修正: 変数名を変更
        merged_lines = merge_lines(detected_lines)  # 線分をマージ
        # long_lines = filter_long_lines(detected_lines)  # 長い線分のみフィルタリング
        # filtered_lines = filter_parallel_white_lines(long_lines, mask)  # 線分フィルタリング
        draw_lines(transformed_image, detected_lines)  # 線分を描画

        cv2.imshow("Original Image", frame)
        cv2.imshow("White Lines Detected", transformed_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
