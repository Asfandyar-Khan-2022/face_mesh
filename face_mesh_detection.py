import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/video8.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
mpFaceMeshConnections =  mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 1)
drawSpec = mpDraw.DrawingSpec(thickness = 5, circle_radius = 5)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMeshConnections,
                                  drawSpec, drawSpec)
            
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                5, (0, 255, 0), 5)

    cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('custom window', img)
    key = cv2.waitKey(1)
    if key > 0:
        break

