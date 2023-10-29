import cv2
import matplotlib.pyplot as plt

# 전역 변수 설정
points = []
count = 0

def select_point(event):
    global points, count, img_temp, ax

    # 왼쪽 마우스 버튼 클릭을 확인
    if event.button == 1:
        count += 1
        x, y = int(event.xdata), int(event.ydata)
        ax.plot(x, y, 'ro')  # 점 표시
        fig.canvas.draw()  # 업데이트

        # 선택한 좌표를 리스트에 추가
        points.append((x, y))

        # 4개의 점을 모두 선택하면 그래프를 닫음
        if count == 4:
            plt.close()

img_path = 'C:/Users/admin/Desktop/imagetest_score/erp.png'
img = cv2.imread(img_path)
img_temp = img.copy()

# 이미지를 RGB 형식으로 변환
img_rgb = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)

# 이미지와 이벤트 핸들러 설정
fig, ax = plt.subplots()
ax.imshow(img_rgb)
fig.canvas.mpl_connect('button_press_event', select_point)
plt.show()

# 선택한 좌표 출력
print(points)

