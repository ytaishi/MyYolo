import cv2
import numpy as np
import math


# 前処理 (ぼかし) 関数
def preprocess_mask(img):
    return cv2.GaussianBlur(img, (11, 11), 0)


# 前処理 (バイラテラル) 関数
# def preprocess_mask(img):
#     return cv2.bilateralFilter(img, 15, sigmaColor=50, sigmaSpace=50)


# threshold on black
def preprocess_color_mask(img):
    lower = (150, 150, 150)
    upper = (255, 255, 255)
    thresh = cv2.inRange(img, lower, upper)

    return thresh


# apply morphology open and close
def apply_morphology(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    return morph


# 長さによって線分をフィルタリングする関数
def filter_long_lines(lines, length_thresh=100):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 線分の長さを計算
        if length > length_thresh:  # 長さが閾値より大きい場合に追加
            filtered_lines.append(line)
    return np.array(filtered_lines)


# 線分を描画する関数
def draw_lines(image, lines):
    line_image = image.copy()
    if lines is not None:
        for line in lines:  # 修正: すでに2次元配列なのでそのままループ
            x0, y0, x1, y1 = map(int, line[0])  # 修正: 最初の4つの要素だけを使用
            cv2.line(line_image, (x0, y0), (x1, y1), (0, 255, 0), 2)

    return line_image


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Fast Line Detectorのオブジェクトを作成
    fld = cv2.ximgproc.createFastLineDetector(
        length_threshold=30,
        distance_threshold=30,
        canny_th1=50,
        canny_th2=50,
        canny_aperture_size=3,
        do_merge=True,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_image = preprocess_mask(frame)  # 前処理 (ぼかし)

        thresh = preprocess_color_mask(blur_image)

        morph = apply_morphology(thresh)

        detected_lines = fld.detect(morph)
        # long_lines = filter_long_lines(detected_lines)  # 長い線分のみフィルタリング
        lined_image = draw_lines(frame, detected_lines)  # 線分を描画

        cv2.imshow("Original image", frame)
        cv2.imshow("Blur image", blur_image)
        cv2.imshow("Threshold image", thresh)
        cv2.imshow("Morphology image", morph)
        cv2.imshow("White Lines Detected", lined_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
