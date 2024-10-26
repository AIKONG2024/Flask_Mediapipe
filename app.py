from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.4, min_tracking_confidence=0.5)

# 이전 프레임의 랜드마크를 저장하기 위한 변수 초기화
previous_landmarks = None
video_width = 720 #비디오 너비
video_height = 480 #비디오 높이
stabilization_factor = 0.4 #현재 프레임의 랜드마크와 이전 프레임의 랜드마크를 혼합하여 안정화하는 데 사용되는 가중치
print_frame_interval = 1  # 몇 프레임마다 랜드마크 좌표를 출력할지 설정

def generate_frames():
    global previous_landmarks
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)  # 해상도를 줄이기 위해 너비 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)  # 해상도를 줄이기 위해 높이 설정
    frame_count = 10
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count += 1

            # 이미지를 좌우 반전하고 BGR을 RGB로 변환
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 전신 랜드마크 검출
            results = pose.process(image)

            # 터미널에 랜드마크 정보 출력 (설정된 프레임 간격마다)
            if frame_count % print_frame_interval == 0 and results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    print(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}, visibility={landmark.visibility}")

            # 이미지를 다시 BGR로 변환하여 표시 준비
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 전신 랜드마크를 그리기 (안정화 처리 추가)
            if results.pose_landmarks:
                current_landmarks = results.pose_landmarks

                # 랜드마크 안정화 처리
                if previous_landmarks:
                    for i in range(len(current_landmarks.landmark)):
                        current_landmarks.landmark[i].x = stabilization_factor * previous_landmarks.landmark[i].x + (1 - stabilization_factor) * current_landmarks.landmark[i].x
                        current_landmarks.landmark[i].y = stabilization_factor * previous_landmarks.landmark[i].y + (1 - stabilization_factor) * current_landmarks.landmark[i].y
                        current_landmarks.landmark[i].z = stabilization_factor * previous_landmarks.landmark[i].z + (1 - stabilization_factor) * current_landmarks.landmark[i].z

                previous_landmarks = current_landmarks

                mp_drawing.draw_landmarks(
                    image, current_landmarks, mp_pose.POSE_CONNECTIONS)

            # JPEG로 인코딩하여 스트리밍
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)