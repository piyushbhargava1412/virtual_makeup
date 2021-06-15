import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

SCALE_FACTOR = 0.5

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # read frame from camera
    ret, img = cam.read()
    if not ret:
        continue

    img_resized = cv2.resize(
        img,
        None,
        fx=SCALE_FACTOR,
        fy=SCALE_FACTOR,
        interpolation=cv2.INTER_LINEAR,
    )

    try:
        # detect the face in the image
        rect = max(detector(img_resized), key=lambda r: r.area())
    except Exception as e:
        print(e)
        continue

    # find landmark points on the detected face
    landmarks = predictor(img_resized, rect)

    # convert dlib points to opencv style points
    points = [(int(point.x / SCALE_FACTOR), int(point.y / SCALE_FACTOR)) for point in landmarks.parts()]

    # use the points to augment makeup on different regions of face
    # draw the points on the input image
    for point in points:
        cv2.circle(img, point, 2, (0, 0, 255), -1)

    # show the output
    cv2.imshow("landmark points", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break