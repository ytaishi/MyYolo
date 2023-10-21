import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread("materials\Screenshot 2023-10-21 12-01-17.png", cv2.IMREAD_COLOR)

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Cannyエッジ検出
edges = cv2.Canny(gray, 100, 150)

# Hough変換で直線を検出
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# 検出した直線を画像に描画
for rho, theta in lines[:, 0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 画像を表示
cv2.imshow("Detected Stop Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
