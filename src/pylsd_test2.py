import cv2
import numpy as np
from pylsd.lsd import lsd


def line_length(x1, y1, x2, y2):
    """線分の長さを計算する関数"""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ウェブカメラの初期化
cap = cv2.VideoCapture(0)

# 最小の線分の長さを設定（この値は調整可能）
min_length = 50

while True:
    ret, frame = cap.read()

    if not ret:
        print("フレームの取得に失敗しました")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    lines = lsd(gray_frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[:4])

            # 線分の長さを計算
            length = line_length(x1, y1, x2, y2)

            # 線分の長さが一定の長さ以上の場合のみ描画
            if length > min_length:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Fast Line Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
