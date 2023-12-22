import cv2
import numpy as np

# 카메라 캘리브레이션 매개변수
mtx = np.array([[466.443413, 0, 320.000000],
                [0, 466.443413, 240.000000],
                [0, 0, 1]], dtype=np.float32)
dist = np.array([0.122349, -0.867988, -0.015407, -0.002095], dtype=np.float32)

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# ORB 특징점 추출기 생성
orb = cv2.ORB_create()

# 이진 기술자를 위한 매처 생성
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

ref_image = None
ref_image_set = False
mode = 0  # 0: 초기 상태, 1: 특징점 매칭 & 3D 좌표 저장, 2: triangulation 및 3D 좌표 출력
ref_rvec, ref_tvec = None, None

while True:
    ret, img = cap.read()
    if not ret:
        break

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == 32:  # Space bar
        if mode == 0:
            ref_image = img.copy()
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            kp1, des1 = orb.detectAndCompute(ref_gray, None)
            ref_image_set = True
            mode = 1
        elif mode == 1:
            mode = 2

    if ref_image_set and mode in [1, 2]:
        # 현재 영상과 레퍼런스 이미지 사이 특징점 매칭
        kp2, des2 = orb.detectAndCompute(gray, None)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        pts0, pts1 = [], []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                pts0.append(kp1[m.queryIdx].pt)
                pts1.append(kp2[m.trainIdx].pt)

        pts0 = np.float32(pts0)
        pts1 = np.float32(pts1)

        if len(pts0) >= 8 and len(pts1) >= 8:
            E, _ = cv2.findEssentialMat(pts0, pts1, focal=mtx[0, 0], pp=(mtx[0, 2], mtx[1, 2]), method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, pts0, pts1, mtx, dist)

            if mode == 1:
                # 매칭 결과 시각화
                img_matches = cv2.drawMatches(ref_image, kp1, img, kp2, [m for m, n in matches if m.distance < 0.75 * n.distance], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow('Feature Matches', img_matches)
            elif mode == 2:
                # triangulation & 3D 좌표 출력
                P0 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
                P1 = np.hstack((R, t))
                X = cv2.triangulatePoints(mtx @ P0, mtx @ P1, pts0.T, pts1.T)
                X /= X[3]
                valid_X = X[:, np.abs(X[3, :]) > 1e-4]

                if valid_X.shape[1] > 0:
                    X_project = valid_X[:3, :].T
                    X_project = X_project.astype(np.float32)
                    projected_pts, _ = cv2.projectPoints(X_project, R, t, mtx, dist)

                    for pt in projected_pts:
                        x, y = int(pt[0][0]), int(pt[0][1])
                        if 0 <= x < width and 0 <= y < height:
                            img = cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

                cv2.imshow('Triangulation and 3D Points', img)
    else:
        cv2.imshow('Live Stream', img)

cv2.destroyAllWindows()
cap.release()
