import cv2
import numpy as np
from pylsd.lsd import lsd


def detect_lines(image_path):
    # 画像をグレースケールで読み込む
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("画像を読み込めませんでした")
        return

    # LSDで直線検出
    lines = lsd(img)

    # 検出した直線を元の画像に描画
    color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[:4])
        cv2.line(color_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 結果を表示
    cv2.imshow("Detected Lines", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 画像パスを指定して関数を実行
    detect_lines("materials\g9pc7kz6.bmp")
