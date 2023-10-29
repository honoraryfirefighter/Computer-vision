import cv2
import numpy as np

def erp_to_top_view(erp_image, W, H, hfov=120, vfov=90):
    # Define focal length based on the ERP image width
    f = W / (2 * np.pi)

    # Calculate dimensions for the top-view projection
    dst_cols = int(2 * f * np.tan(np.radians(hfov) / 2) + 0.5)
    dst_rows = int(2 * f * np.tan(np.radians(vfov) / 2) + 0.5)

    # Create an empty image for the top-view projection
    top_view = np.zeros((dst_rows, dst_cols, 3), dtype=np.uint8)

    # Center coordinates for the top-view projection
    dst_cx = dst_cols // 2
    dst_cy = dst_rows // 2

    # Generate the top-view projection
    for x in range(dst_cols):
        xth = np.arctan((x - dst_cx) / f)
        src_x = int((xth) * W / (2 * np.pi) + 0.5) % W

        yf = f / np.cos(xth)
        for y in range(dst_rows):
            yth = np.arctan((y - dst_cy) / yf)
            src_y = int(yth * H / np.pi + H / 2 + 0.5) % H

            top_view[y, x] = erp_image[src_y, src_x]

    return top_view

# Example usage
erp_image = cv2.imread(r'C:/Users/admin/Desktop/imagetest_score/erp.png')  # r을 사용하여 raw string으로 정의
W, H = erp_image.shape[1], erp_image.shape[0]
top_view = erp_to_top_view(erp_image, W, H)
cv2.imwrite('C:/Users/admin/Desktop/imagetest_score/top_view_output2.jpg', top_view)

