import cv2
import numpy as np

# 카메라 캘리브레이션 매개변수
mtx = np.array([[466.443413, 0, 320.000000],
                [0, 466.443413, 240.000000],
                [0, 0, 1]], dtype=np.float32)
dist = np.array([0.122349, -0.867988, -0.015407, -0.002095], dtype=np.float32)

# 초기화
image_db = []       
features_db = []    
coords_db = []      
current_pose = np.eye(4)  

# 웹캠 초기화
cap = cv2.VideoCapture(0)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# 메인 루프
while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None)

    img_display = img.copy()

    best_match_image = None
    best_match_features = None
    best_match_pts1 = None
    best_match_pts2 = None
    best_match_pose = np.eye(4)

    for i, (ref_kp, ref_des) in enumerate(features_db):
        matches = bf.knnMatch(ref_des, des2, k=2)
        good_matches = []
        pts0 = []
        pts1 = []
        
        # Lowe's ratio test
        ratio_thresh = 0.6
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                pts0.append(ref_kp[m.queryIdx].pt)
                pts1.append(kp2[m.trainIdx].pt)

        pts0 = np.float32(pts0)
        pts1 = np.float32(pts1)

        if len(good_matches) >= 8:  
            E, mask = cv2.findEssentialMat(pts0, pts1, focal=mtx[0, 0], pp=(mtx[0, 2], mtx[1, 2]), method=cv2.RANSAC, prob=0.999, threshold=0.5)  
            _, R, t, mask = cv2.recoverPose(E, pts0, pts1, mtx, dist, mask=mask)

            t_matrix = np.hstack((R, t))
            t_matrix = np.vstack((t_matrix, [0, 0, 0, 1]))
            current_pose = current_pose @ t_matrix

            if best_match_image is None or len(pts0) > len(best_match_pts1):
                best_match_image = image_db[i]
                best_match_features = (ref_kp, ref_des)
                best_match_pts1 = pts0
                best_match_pts2 = pts1
                best_match_pose = current_pose

            
            if best_match_image is not None:
                P0 = np.eye(3, 4)
                P1 = best_match_pose[:3, :4]
                X = cv2.triangulatePoints(P0, P1, np.float32(best_match_pts1).T, np.float32(best_match_pts2).T)
                X /= X[3]
                coords_db.append(X)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # 스페이스바
        print("Spacebar pressed")
        image_db.append(img)
        kp, des = sift.detectAndCompute(gray, None)
        features_db.append((kp, des))
        print(f"Image DB size: {len(image_db)}, Features DB size: {len(features_db)}")
    elif key == 27:  # ESC
        break

    
    if len(features_db) > 0:
        for kp in features_db[-1][0]:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)

    
    for X in coords_db:
        projected_pts, _ = cv2.projectPoints(X[:3, :], current_pose[:3, :3], current_pose[:3, 3], mtx, dist)
        for pt in projected_pts:
            x, y = int(pt[0][0]), int(pt[0][1])
            if 0 <= x < img_display.shape[1] and 0 <= y < img_display.shape[0]:
                cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Visual SLAM', img_display)

cv2.destroyAllWindows()
cap.release()

