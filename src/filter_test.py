import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread("materials\Screenshot 2023-10-21 13-24-59.png", 0)

# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Median Filtering
median_filter = cv2.medianBlur(image, 5)

# Bilateral Filtering
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

# Non-Local Means
non_local_means = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)

# Total Variation Denoising (OpenCVには直接的な関数はないが、実装可能)

# 画像の表示（ここではGaussian Blurの例）
cv2.imshow("Original", image)
cv2.imshow("Gaussian Blur", gaussian_blur)
cv2.imshow("Median Filter", median_filter)
cv2.imshow("Bilateral Filter", bilateral_filter)
cv2.imshow("Non-Local Means", non_local_means)

cv2.waitKey(0)
cv2.destroyAllWindows()
