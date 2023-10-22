import cv2
import numpy as np


# 画像を読み込む関数
def read_image(filename):
    return cv2.imread(filename)


# 射影変換を行う関数
def perspective_transform(image):
    src = np.float32([[0 + 400, 0 + 000], [1280 - 400, 0 + 000], [0, 720], [1280, 720]])
    dst = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))


# 白色を検出する関数
def detect_white(image):
    # BGRからHSVへ変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 白色を検出するための閾値
    lower_white = np.array([0, 0, 100], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)

    # 白色のみ抽出
    mask = cv2.inRange(hsv, lower_white, upper_white)

    return mask


# ぼかしフィルタを適用する関数
def apply_blur(image):
    # ガウシアンぼかしを適用
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred_image


# 白線を検出する関数
def detect_white_lines(image):
    # ぼかしフィルタを適用
    blurred_image = apply_blur(image)

    # 白色を検出
    mask = detect_white(blurred_image)

    # Cannyエッジ検出
    canny = cv2.Canny(mask, 150, 300)

    # Hough変換で直線検出
    lines = cv2.HoughLinesP(
        canny, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=20
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if -10 <= angle <= 10:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return image


# if __name__ == "__main__":
#     image = read_image("materials\Screenshot 2023-10-21 13-24-59.png")
#     cv2.imshow("Original Image", image)

#     transformed_image = perspective_transform(image)
#     cv2.imshow("Transformed Image", transformed_image)

#     white_line_image = detect_white_lines(transformed_image)
#     cv2.imshow("White Lines Detected", white_line_image)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    # Webカメラのキャプチャを開始
    cap = cv2.VideoCapture(0)

    # カメラの解像度を1280x720に設定
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        # カメラからフレームを取得
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # 射影変換を行う
        transformed_image = perspective_transform(frame)

        # 白線を検出する
        white_line_image = detect_white_lines(transformed_image)

        # オリジナルの画像と、検出後の画像を表示
        cv2.imshow("Original Image", frame)
        cv2.imshow("White Lines Detected", white_line_image)

        # 'q'キーが押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # キャプチャをリリースして、ウィンドウを全て閉じる
    cap.release()
    cv2.destroyAllWindows()
