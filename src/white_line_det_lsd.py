import cv2
import numpy as np
import math


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
    upper_white = np.array([180, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower_white, upper_white)


# 前処理 (ぼかし) 関数
def preprocess_mask(mask):
    return cv2.GaussianBlur(mask, (15, 15), 0)


# 線分検出 (LSD) 関数
def detect_lines(mask):
    lsd = cv2.createLineSegmentDetector(0)
    print(lsd.detect(mask))
    return lsd.detect(mask)


# 線分の角度が近似しているかを確認
def are_angles_similar(angle1, angle2, threshold=10):
    return abs(angle1 - angle2) < threshold


# 平行な線分を見つける
def find_parallel_lines(lines, angle_threshold=10):
    parallel_lines = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            angle1 = (
                math.atan2(lines[i][3] - lines[i][1], lines[i][2] - lines[i][0])
                * 180.0
                / math.pi
            )
            angle2 = (
                math.atan2(lines[j][3] - lines[j][1], lines[j][2] - lines[j][0])
                * 180.0
                / math.pi
            )
            if are_angles_similar(angle1, angle2, angle_threshold):
                parallel_lines.append((lines[i], lines[j]))
    return parallel_lines


# 線分の間隔が指定範囲内か確認
def is_within_distance(line1, line2, min_distance=30, max_distance=100):
    # 線分の中点を計算
    mid1 = [(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2]
    mid2 = [(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2]
    distance = math.sqrt((mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2)
    return min_distance <= distance <= max_distance


# 平行な線分の内部が白に近いか確認
def is_inside_white(image, line1, line2):
    # 線分の中点を計算
    mid1 = [(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2]
    mid2 = [(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2]
    # 中点間の中点を計算
    mid = [(mid1[0] + mid2[0]) / 2, (mid1[1] + mid2[1]) / 2]
    mid = list(map(int, mid))
    # 画像のHSV値を取得
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixel = hsv[mid[1], mid[0]]
    return 100 <= pixel[2] <= 255  # V（明度）が高い場合に白とみなす


# 線分のフィルタリング関数
def filter_lines(image, lines):
    filtered_lines = []
    if lines is not None:
        # 形状を(n, 4)に変更
        lines = lines.squeeze()
        for line1, line2 in find_parallel_lines(lines):
            if is_within_distance(line1, line2) and is_inside_white(image, line1, line2):
                filtered_lines.append(line1)
                filtered_lines.append(line2)
    return filtered_lines


# 線分を描画する関数
def draw_lines(image, lines):
    for line in lines:
        x0, y0, x1, y1 = line
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 2)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 射影変換
        transformed_image = perspective_transform(frame)
        # 白色を検出
        mask = detect_white(transformed_image)
        # 前処理 (ぼかし)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        # 線分検出 (LSD)
        lsd = cv2.createLineSegmentDetector(0)
        lines, _width, _prec, _nfa = lsd.detect(mask)
        # 線分のフィルタリング
        filtered_lines = filter_lines(transformed_image, lines)
        # 線分の描画
        draw_lines(transformed_image, filtered_lines)
        # 画像の表示
        cv2.imshow("Original Image", frame)
        cv2.imshow("White Lines Detected", transformed_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
