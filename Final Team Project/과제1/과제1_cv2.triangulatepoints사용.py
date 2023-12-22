import cv2
import numpy as np

def insteadfindFundamentalMat(pts1, pts2):
    if len(pts1) != len(pts2):
        raise ValueError("Point arrays 길이는 같아야 한다.")
    
    pts1 = np.hstack((pts1, np.ones((len(pts1), 1))))
    pts2 = np.hstack((pts2, np.ones((len(pts2), 1))))

    A = []
    for p1, p2 in zip(pts1, pts2):
        x1, y1, _ = p1
        x2, y2, _ = p2
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt
    return F / F[-1, -1]

def insteadfindEssentialMat(pts1, pts2, K):
    F = insteadfindFundamentalMat(pts1, pts2)
    return K.T @ F @ K

CHECKERBOARD = (4, 4)
mtx = np.array([[466.443413, 0, 320.000000],
                [0, 466.443413, 240.000000],
                [0, 0, 1]], dtype=np.float32)
dist = np.array([0.122349, -0.867988, -0.015407, -0.002095], dtype=np.float32)

cap = cv2.VideoCapture(0)
ref_image, ref_image_set, mode = None, False, 0
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
ref_rvec, ref_tvec = None, None

while True:
    ret, img = cap.read()
    if not ret:
        break

    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        if mode == 0:
            ref_image = img.copy()
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            kp1, des1 = sift.detectAndCompute(ref_gray, None)
            ref_image_set = True

            ret, corners = cv2.findChessboardCorners(ref_gray, CHECKERBOARD, None)
            if ret:
                corners2 = cv2.cornerSubPix(ref_gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                ret, ref_rvec, ref_tvec = cv2.solvePnP(objp, corners2, mtx, dist)
            mode = 1
        elif mode == 1:
            mode = 2

    if ref_image_set and mode in [1, 2]:
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(frame_gray, None)
        matches = bf.knnMatch(des1, des2, k=2)

        pts0, pts1 = [], []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                pts0.append(kp1[m.queryIdx].pt)
                pts1.append(kp2[m.trainIdx].pt)

        pts0 = np.float32([kp1[m.queryIdx].pt for m, n in matches if m.distance < 0.75 * n.distance])
        pts1 = np.float32([kp2[m.trainIdx].pt for m, n in matches if m.distance < 0.75 * n.distance])

        if len(pts0) >= 5 and len(pts1) >= 5:
            E = insteadfindEssentialMat(pts0, pts1, mtx)
            _, R, t, mask = cv2.recoverPose(E, pts0, pts1, mtx)

            if mode == 1:
                img_matches = cv2.drawMatches(ref_image, kp1, img, kp2, [m for m, n in matches if m.distance < 0.75 * n.distance], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow('Feature Matches', img_matches)
            elif mode == 2:
                P0 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
                P1 = np.hstack((R, t))
                X = cv2.triangulatePoints(mtx @ P0, mtx @ P1, pts0.T, pts1.T)  # OpenCV triangulatePoints 사용
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
