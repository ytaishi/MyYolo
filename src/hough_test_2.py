import cv2
import numpy as np

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラが開けません")
    exit()

while True:
    # カメラからフレームを読み込む
    ret, frame = cap.read()
    if not ret:
        print("フレームが読み込めません")
        break

    # フレームにぼかしフィルタを適用
    blurred_frame = cv2.GaussianBlur(frame, (9, 9), 0)

    # 適応的二値化処理
    gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    binary_frame = cv2.adaptiveThreshold(
        gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # ハフ変換による直線検出
    lines = cv2.HoughLinesP(
        binary_frame, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10
    )

    # 描画用の画像を準備
    line_image = np.copy(frame) * 0  # 元の画像と同じサイズの黒い画像を作成

    # 検出された直線を描画
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > 100:  # 短い直線を除去する条件
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 元の画像に直線を描画
    lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # 途中結果と最終結果を表示
    cv2.imshow("Binary Frame", binary_frame)  # 二値化画像
    cv2.imshow("Lines Detection", line_image)  # 検出された直線
    cv2.imshow("Final Result", lines_edges)  # 最終結果

    # 'q'を押すとループを抜ける
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# カメラを解放
cap.release()

# すべてのウィンドウを破棄
cv2.destroyAllWindows()
