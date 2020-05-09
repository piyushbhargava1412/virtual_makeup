#!/usr/bin/env python

# import the necessary packages
from collections import OrderedDict
import numpy as np
import dlib
import cv2
from scipy.interpolate import interp1d

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor('../models/dlib/shape_predictor_68_face_landmarks.dat')
lipstick_color = None
shadow_color = None
setliner = False

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 67)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

LIP_LANDMARKS = {
    "outer_lip": list(range(48, 59)),
    "inner_lip": list(range(60, 67)),
    "lip_corners": [48, 54, 51, 57]
}

EYE_LANDMARKS = {
    "left_eye_all": [36, 37, 38, 39, 40, 41],
    "left_eye_lower": [36, 41, 40, 39],
    "left_eye_upper": [36, 37, 38, 39],
    "right_eye_all": [42, 43, 44, 45, 46, 47],
    "right_eye_lower": [42, 47, 46, 45],
    "right_eye_upper": [42, 43, 44, 45]
}


def on_mouse_click(event, x, y, flags, frame):
    global lipstick_color
    global shadow_color
    global setliner
    if x >= 640:
        if event == cv2.EVENT_LBUTTONDBLCLK:
            setliner = not setliner
        return
    elif event == cv2.EVENT_LBUTTONUP:
#         if y <= 280:
            lipstick_color = frame[y, x].tolist()
#         else:
#             shadow_color = frame[y, x].tolist()
    elif event == cv2.EVENT_MBUTTONUP:
        lipstick_color = None
        shadow_color = None
        setliner = False

def resetApp():
    global lipstick_color
    global shadow_color
    global setliner
    lipstick_color = None
    shadow_color = None
    setliner = False

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def dlib_rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def get_face_landmarks(image, detected_face):
    """
    Encode face into 128 measurements using a neural net
    :param image: picture numpy array
    :param detected_face: face detector object with one detected face
    :return: array of (x,y) tuple coordinates
    """
    shape = face_pose_predictor(image, detected_face)
    return shape_to_np(shape)


def detect_face_dlib(image):
    detected_faces = face_detector(image, 0)

    # if faces were not found, try finding faces in upsampled the image
    if detected_faces is None:
        # Run the HOG face detector on the image data.
        # The result will be the bounding boxes of the faces in our image.
        detected_faces = face_detector(image, 1)

    return detected_faces


def visualize_facial_landmarks(image, shape, output, drawtype="fill", colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        # colors = [(19, 199, 109), (79, 76, 250), (250, 159, 23),
        #           (168, 100, 250), (250, 163, 32),
        #           (163, 38, 32), (180, 42, 250)]
        colors = [(0, 255, 0), (0, 255, 0), (0, 255, 0),
                  (0, 255, 0), (0, 255, 0),
                  (0, 255, 0), (0, 255, 0)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        if drawtype == "fill":
            # check if are supposed to draw the jawline
            if name == "jaw":
                # since the jawline is a non-enclosed facial region,
                # just draw lines between the (x, y)-coordinates
                for l in range(1, len(pts)):
                    ptA = tuple(pts[l - 1])
                    ptB = tuple(pts[l])
                    cv2.line(overlay, ptA, ptB, colors[i], 3)

            # otherwise, compute the convex hull of the facial
            # landmark coordinates points and display it
            else:
                hull = cv2.convexHull(pts)
                cv2.drawContours(overlay, [hull], -1, colors[i], -1)
        else:
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # # return the output image
    # return output


def detectLip(lmPts):
    olPts = LIP_LANDMARKS["outer_lip"]
    ilPts = LIP_LANDMARKS["inner_lip"]
    xmargin = 5
    ymargin = 5

    # CREATE BOUNDING RECTANGLE
    x, y = int(lmPts[48][0] - xmargin), int(lmPts[51][1] - ymargin)
    width, height = lmPts[54][0] - lmPts[48][0] + 2 * xmargin, lmPts[57][1] - lmPts[51][1] + 2 * ymargin
    # print ("RECT " ,x,y,width,height)

    # Outer Mask
    lineMask = np.zeros((height, width), dtype=np.uint8)
    for i in olPts:
        x1, y1 = lmPts[i + 1][0], lmPts[i + 1][1]
        x2, y2 = lmPts[i][0], lmPts[i][1]
        cv2.line(lineMask, (x1 - x, y1 - y), (x2 - x, y2 - y), (255, 0, 0), 1)
    x1, y1 = lmPts[olPts[0]][0], lmPts[olPts[0]][1]
    x2, y2 = lmPts[59][0], lmPts[59][1]
    cv2.line(lineMask, (x1 - x, y1 - y), (x2 - x, y2 - y), (255, 0, 0), 1)

    _, contours, _ = cv2.findContours(lineMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lineMask = cv2.drawContours(lineMask, contours, -1, (255), -1)

    # Innner Mask
    innerMask = np.zeros((height, width), dtype=np.uint8)
    for i in ilPts:
        x1, y1 = lmPts[i + 1][0], lmPts[i + 1][1]
        x2, y2 = lmPts[i][0], lmPts[i][1]
        cv2.line(innerMask, (x1 - x, y1 - y), (x2 - x, y2 - y), (255, 0, 0), 1)
    x1, y1 = lmPts[ilPts[0]][0], lmPts[ilPts[0]][1]
    x2, y2 = lmPts[67][0], lmPts[67][1]
    cv2.line(innerMask, (x1 - x, y1 - y), (x2 - x, y2 - y), (255, 0, 0), 1)
    _, contours, _ = cv2.findContours(innerMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    innerMask = cv2.drawContours(innerMask, contours, -1, (255), -1)

    # Subtract from outer mask
    cv2.subtract(lineMask, innerMask, lineMask)
    # cv2.imshow("mask", lineMask)
    rect = [x, y, width, height]
    return lineMask, rect


def addColor(cropped_img, color):
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    tempImg = hsv_img.copy()

    # color = (150,255,139)

    # create a 1x1 3 channel image
    color = np.uint8([[[color[0], color[1], color[2]]]])

    # convert the RGB image to HSV and take the first pixel
    color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]

    # create a tuple of the HSV color
    color = (int(color[0]), int(color[1]), int(color[2]))

    rows, cols, _ = hsv_img.shape
    for i in range(rows):
        for j in range(cols):
            h, s, v = tempImg[i][j]
            # replacing the hue
            h = color[0]
            diff_s = color[1] - s
            diff_v = color[2] - v
            diff_s, diff_v = diff_s / 1.8, diff_v / 1.3
            s, v = color[1] - diff_s, color[2] - diff_v
            tempImg[i][j] = [h, s, v]

    finalImg = cv2.cvtColor(tempImg, cv2.COLOR_HSV2BGR)
    # print(finalImg.shape)
    # finalImg = cv2.bitwise_and(finalImg,cv2.merge((lipMask,lipMask,lipMask)))
    return finalImg


def applyLipstick(image, mask, bbox, color):
    # get the bounding box coordinates
    lx, ly, lw, lh = bbox

    # extract the lip region from the input image
    image_copy = image.copy()
    lip_image = image[ly: ly + lh, lx: lx + lw]

    # create a tuple from RGB list
    color = (int(color[0]), int(color[1]), int(color[2]))

    # add the lipstick color to the bounding box in HSV space to retain texture and only
    # replace the Hue
    lip_image_hue_modified = addColor(lip_image, color)
    # cv2.imshow("added", lip_image_hue_modified)

    mask = cv2.blur(mask, (5, 5))
    for row in range(lh):
        for col in range(lw):
            if mask[row, col] != 0:
                alpha = (mask[row, col] * 1.0) / 255
                lip_image[row, col] = alpha * lip_image_hue_modified[row][col] + (1 - alpha) * lip_image[row, col]

    # cv2.imshow("Lipstick applied", lip_image)
    # cv2.waitKey(10)
    image_copy[ly:ly + lh, lx: lx + lw] = lip_image
    return image_copy


def setEyeliner(frame, lmPoints, color):
    frame_copy = frame.copy()

    mask1 = _retMask(frame, lmPoints, EYE_LANDMARKS["left_eye_upper"], 2)
    mask2 = _retMask(frame, lmPoints, EYE_LANDMARKS["left_eye_lower"], 1, (0, 1))
    mask3 = _retMask(frame, lmPoints, EYE_LANDMARKS["right_eye_upper"], 2)
    mask4 = _retMask(frame, lmPoints, EYE_LANDMARKS["right_eye_lower"], 1, (0, 1))

    # Apply bitwise_or operation to combine all mask
    mask = mask1 | mask3 | mask2 | mask4

    # Apply blur operation to blend mask at the edges , to make it look natural by blending
    mask = cv2.blur(mask, (3, 3))

    # HSV blending
    # _fillColor(frame_copy, mask, color)

    # Color of eye liner ( BLACK )
    liner_color = np.array([color[0], color[1], color[2]], dtype=np.uint8)

    # Apply blending equation
    # I = apha*F + (1-alpha)*B

    h, w, _ = frame.shape
    for row in range(h):
        for col in range(w):
            if mask[row, col]:
                alpha = (mask[row, col] * 1.0) / 255
                frame_copy[row, col] = alpha * liner_color + (1 - alpha) * frame[row, col]

    # return the image with applied Eye Liner
    return frame_copy


def _retMask(frame, points, feature_pts, thickness=1, shift=(0, 0)):
    """ return a mask joining input points of given thickness"""
    # frameCopy = frame.copy()

    # create Mask
    w, h, _ = frame.shape
    mask = np.zeros((w, h), np.uint8)

    # interpolate points
    points = _interpolate(points, feature_pts)

    # shift the points
    shift_x, shift_y = shift
    if shift_x != 0 or shift_y != 0:
        for p in points:
            p[0] += shift_x
            p[1] += shift_y

    # draw line on Mask
    x1, y1 = points[0]
    for p in points[1:]:
        x2, y2 = p
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_8)
        x1, y1 = x2, y2

    return mask


def _interpolate(points, feature_pts):
    "Interpolate the input points to fit on a smooth curve "

    N = 6  # Number of output points

    points1 = [points[i] for i in feature_pts]
    x = np.array(points1)[:, 0]
    y = np.array(points1)[:, 1]

    f2 = interp1d(x, y, kind="cubic")
    xnew = np.linspace(x[0], x[-1], num=N, endpoint=True)

    newPoints = []
    for x, y in zip(xnew, f2(xnew)):
        newPoints.append([int(x), int(y)])
    # newPoints = [ [int(round(x,0)),int(round(y))] for x,y in zip(xnew,f2(xnew)) ]
    return newPoints


"""------------------Functions for setting eyeShadow-------------------------"""


def setEyeShadow(frame, lmpoints, color):
    frame_copy = frame.copy()

    # Create mask
    thickness = int(round((lmpoints[41][1] - lmpoints[37][1]) / 2))

    mask1 = _eyeShadowMask(frame, lmpoints, EYE_LANDMARKS["left_eye_upper"], thickness, shift=(0, -4))
    mask2 = _eyeShadowMask(frame, lmpoints, EYE_LANDMARKS["right_eye_upper"], thickness, shift=(0, -4))

    # Apply bitwise_or operation to combine mask of left and right eye
    mask = mask1 | mask2
    mask = cv2.blur(mask, (5, 5))

    # Fill shadow color
    _fillColor(frame_copy, mask, color)

    return frame_copy


def _fillColor(frame, mask, color):
    """ Apply the color"""
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color = np.uint8([[[color[0], color[1], color[2]]]])
    color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
    color = (int(color[0]), int(color[1]), int(color[2]))

    rows, cols, _ = frame.shape

    for i in range(rows):
        for j in range(cols):
            if mask[i, j]:
                h, s, v = hsvImg[i][j]
                h = color[0]
                diff_s = color[1] - s
                diff_v = color[2] - v
                diff_s, diff_v = diff_s / 1.6, diff_v / 1.3
                s, v = color[1] - diff_s, color[2] - diff_v
                hsvImg[i][j] = [h, s, v]

                # fill color on BGR image
                hsv = np.uint8([[[h, s, v]]])
                temp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                alpha = (mask[i, j] * 1.0) / 255
                frame[i, j] = alpha * temp + (1 - alpha) * frame[i, j]

    # tempImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    # #fill color with mask
    # for row in range(rows):
    #     for col in range(cols):
    #         if mask[row, col]:
    #             alpha = (mask[row, col] * 1.0) / 255
    #             frame[row, col] = alpha * tempImg[row][col] + (1 - alpha) * frame[row, col]


def _eyeShadowMask(frame, points, feature_pts, thickness=1, shift=(0, 0)):
    # create Mask
    w, h, _ = frame.shape
    mask = np.zeros((w, h), np.uint8)

    # interpolate points
    points = _interpolate(points, feature_pts)

    # shift the points
    shift_x, shift_y = shift
    if shift_x != 0 or shift_y != 0:
        for p in points:
            p[0] += shift_x
            p[1] += shift_y

    # increase width of mask
    points[0][0] -= 6

    # generate mask
    l_points = points
    u_points = [(p[0], p[1] - thickness) for p in points]
    u_points.reverse()
    all_points = np.array(l_points + u_points)

    # fill the mask
    # cv2.fillPoly(mask,[all_points],(255))
    cv2.drawContours(mask, [all_points], -1, 255, -1)

    return mask


def process_webcam():
    cv2.namedWindow("Output")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    shade_chart = cv2.imread('./lipstick_products.jpg', 1)
    print(shade_chart.shape[0], shade_chart.shape[1])
    final_img = np.zeros((760, 640, 3), np.uint8)
    final_img[480:shade_chart.shape[0]+480, :shade_chart.shape[1]] = shade_chart
#     cv2.imshow("Output", final_img)
#     cv2.waitKey(0)

    cv2.setMouseCallback('Output', on_mouse_click, final_img)
    # cv2.waitKey(10)

    while True:
        # Capture frame-by-frame
        ret, image = cap.read()
        rects = detect_face_dlib(image)

        output = image.copy()
        # visual = image.copy()
        for rect in rects:
            landmark_pts = get_face_landmarks(image, rect)

            # Get the lip mask
            mask, bbox = detectLip(landmark_pts)
            if lipstick_color is not None:
                # Apply the lipstick
                output = applyLipstick(output, mask, bbox, lipstick_color)

            if setliner is True:
                output = setEyeliner(output, landmark_pts, [0, 0, 0])

            if shadow_color is not None:
                output = setEyeShadow(output, landmark_pts, shadow_color)

            # visualize all facial landmarks with a transparent overlay
            # visualize_facial_landmarks(image, landmark_pts, visual)

            # draw bounding boxes
            # (x, y, w, h) = dlib_rect_to_bb(rect)
            # cv2.rectangle(visual, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # get the resultant image
        final_img[:output.shape[0], :output.shape[1]] = output

        cv2.imshow("Output", final_img)
        # cv2.imshow("Visual", visual)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('r'):
            resetApp()
        elif key & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()


def plot_landmark_points(output, landmark_points):
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in landmark_points:
        cv2.circle(output, (x, y), 3, (0, 0, 255), -1)


def process_image(filename):
    im = cv2.imread(filename, 1)
    if im is None:
        print("Unable to read Image")
        return

    image = cv2.resize(im, (480, 640))
    output = image.copy()

    # Capture frame-by-frame
    rects = detect_face_dlib(image)

    for rect in rects:
        landmark_pts = get_face_landmarks(image, rect)

        # Get the lip mask
        mask, bbox = detectLip(landmark_pts)
        cv2.imshow("Mask", mask)

        # visualize all facial landmarks with a transparent overlay
        visualize_facial_landmarks(image, landmark_pts, output, "line")
        # plot_landmark_points(output, landmark_pts)

        # draw bounding boxes
        (x, y, w, h) = dlib_rect_to_bb(rect)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Output", output)
    cv2.waitKey(0)


if __name__ == "__main__":
    process_webcam()
    # process_image('./PBFace.jpg')
    cv2.destroyAllWindows()
