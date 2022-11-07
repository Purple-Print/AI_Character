#얼굴 추출
import mediapipe as mp
import cv2
contour_list = [152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color = (0,255,0))
lineDrawingSpec = mp_drawing.DrawingSpec(thickness=1, color=(0,255,0))
point_list=[]
with mp_face_mesh.FaceMesh(static_image_mode=True,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5) as face_mesh:
    image = cv2.imread('dataset/testing_set/Oblong/oblong (6).jpg')
                    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
        pass

    annotated_image = image.copy()
    # 얼굴형 -> 채워진 도형
    mesh_dot = results.multi_face_landmarks[0].landmark
    for i in contour_list:
        point = mesh_dot[i]
        print(point)