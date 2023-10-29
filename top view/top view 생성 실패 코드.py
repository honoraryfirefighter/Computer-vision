import cv2
import numpy as np
import matplotlib.pyplot as plt

# ERP 이미지를 정면 이미지로 변환
def erp_to_front_view(erp, W, H, theta, phi, fov):
    f = W / (2 * np.pi)
    R = f / np.tan(fov / 2)

    W_out = int(2 * R * np.tan(fov / 2))
    H_out = W_out

    front_view = np.zeros((H_out, W_out, 3), dtype=np.uint8)

    cx, cy = W_out // 2, H_out // 2

    for x in range(W_out):
        for y in range(H_out):
            x_theta = (x - cx) * fov / W_out
            y_phi = (y - cy) * fov / H_out

            x_s = -R * np.sin(x_theta)
            y_s = R * np.sin(y_phi)
            z_s = R * np.cos(x_theta) * np.cos(y_phi)

            x_e = np.cos(theta) * np.cos(phi)
            y_e = np.sin(theta) * np.cos(phi)
            z_e = np.sin(phi)

            scalar = x_s * x_e + y_s * y_e + z_s * z_e
            if scalar > 0:
                erp_x = (theta + np.arctan2(y_s, x_s)) * W / (2 * np.pi)
                erp_y = (np.pi / 2 - np.arccos(z_s / R)) * H / np.pi

                if 0 <= erp_x < W and 0 <= erp_y < H:
                    front_view[y, x] = erp[int(erp_y), int(erp_x)]

    return front_view

# persepective transform
def perspective_transform(image, src_points):
    dst_points = np.array([
        [0, 0],
        [image.shape[1] - 1, 0],
        [image.shape[1] - 1, image.shape[0] - 1],
        [0, image.shape[0] - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    return warped

erp_image = cv2.imread('C:/Users/admin/Desktop/imagetest_score/erp.png')
W, H = erp_image.shape[1], erp_image.shape[0]

theta = 0
phi = 0
fov = np.pi/2  # 90도

front_view = erp_to_front_view(erp_image, W, H, theta, phi, fov)

src_points = np.array([
    [903, 506],
    [997, 508],
    [839, 549],
    [1061, 554]
], dtype=np.float32)

top_view = perspective_transform(front_view, src_points)

# 원본 이미지에서 4개의 점을 표시
for pt in src_points:
    cv2.circle(erp_image, tuple(pt.astype(int)), 1, (0, 0, 255), -1)

plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(erp_image, cv2.COLOR_BGR2RGB))
plt.title('Original ERP Image')
plt.axis('off')

# Top View 이미지 표시
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(top_view, cv2.COLOR_BGR2RGB))
plt.title('Top View')
plt.axis('off')

plt.tight_layout()
plt.show()



