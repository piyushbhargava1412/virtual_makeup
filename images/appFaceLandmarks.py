# import the necessary packages
from collections import OrderedDict
import numpy as np
import dlib
import cv2
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.colorchooser import askcolor

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor('../models/dlib/shape_predictor_68_face_landmarks.dat')

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
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

# For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS


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


def visualize_facial_landmarks(image, shape, output, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        # check if are supposed to draw the jawline
        if name == "jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # # return the output image
    # return output


def detectLip(image, lmPts):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    return (lineMask, rect)


def addColor(maskedHSV, color):
    maskedHSV = cv2.cvtColor(maskedHSV, cv2.COLOR_BGR2HSV)
    tempImg = maskedHSV.copy()

    # color = (150,255,139)
    color = np.uint8([[[color[0], color[1], color[2]]]])
    color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
    color = (int(color[0]), int(color[1]), int(color[2]))

    rows, cols, _ = maskedHSV.shape
    for i in range(rows):
        for j in range(cols):
            h, s, v = tempImg[i][j]
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
    lx, ly, lw, lh = bbox
    img1 = image.copy()
    im = img1[ly: ly + lh, lx: lx + lw]

    maskColor = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    maskColor[:] = colors[color]
    added = addColor(im, colors[color])

    mask = cv2.blur(mask, (5, 5))
    for row in range(lh):
        for col in range(lw):
            if mask[row, col] or 1:
                alpha = (mask[row, col] * 1.0) / 255
                maskColor[row, col] = alpha * added[row][col] + (1 - alpha) * im[row, col]
            else:
                maskColor[row, col] = im[row, col]

    img1[ly:ly + lh, lx: lx + lw] = maskColor
    return img1


def webcam_run():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while (not exitFlag):
        # Capture frame-by-frame
        ret, image = cap.read()
        rects = detect_face_dlib(image)
        lipInfo = {"mask": None, "bbox": None}

        output = image.copy()
        for rect in rects:
            landmark_pts = get_face_landmarks(image, rect)

            # Get the lip mask
            mask, bbox = detectLip(image, landmark_pts)
            lipInfo["mask"] = mask
            lipInfo["bbox"] = bbox

            # Apply the lipstick
            output = applyLipstick(image, mask, bbox, lipstick_color)

            # get the resultant image

            # visualize all facial landmarks with a transparent overlay
            # visualize_facial_landmarks(image, landmark_pts, output)

            # draw bounding boxes
            (x, y, w, h) = dlib_rect_to_bb(rect)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Image", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        root.update()

    # When everything done, release the capture
    cap.release()


def changeColor(color):
    global lipstick_color
    selected = color


def selectColor():
    global colors
    color, _ = askcolor()
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    colors["selected"] = (b, g, r)


def on_quit():
    global exitFlag
    exitFlag = True
    root.destroy()


#
# if __name__ == "__main__":
#     webcam_run()
#     cv2.destroyAllWindows()


# gui Window
root = tk.Tk()

# # blit LOGO
# logoLabel = Label(root)
# logo = Image.open("kritikal.png")
# logo = logo.resize((125,80))
# logo = ImageTk.PhotoImage(logo)
# logoLabel.config(image = logo)
# logoLabel.pack()
# timeLabel = Label(root, text="Time:")
# timeLabel.pack()

selectColor = ttk.Button(root, text="Choose Color", command=selectColor)
start = ttk.Button(root, text="Start", command=webcam_run)

# ----------------------PACK COLORS -----------------------
colorFrame = tk.Frame(root)
red = tk.Button(colorFrame, bg="red", highlightbackground="red", command=lambda: changeColor("red"), width=2, height=1)
red.bind('<Button-1>', changeColor("red"))
pink = tk.Button(colorFrame, highlightbackground="pink", command=lambda: changeColor("pink"), width=2, height=1)
blue = tk.Button(colorFrame, highlightbackground="blue", command=lambda: changeColor("blue"), width=2, height=1)
red2 = tk.Button(colorFrame, highlightbackground="red2", command=lambda: changeColor("red2"), width=2, height=1)
wheat = tk.Button(colorFrame, highlightbackground="wheat", command=lambda: changeColor("wheat"), width=2, height=1)

# red = tk.Button(colorFrame, bg="#ff0000", command=lambda: changeColor("red"), width=2, height=1)
# pink = tk.Button(colorFrame, bg="#ff00ff", command=lambda: changeColor("pink"), width=2, height=1)
# blue = tk.Button(colorFrame, bg="#722B71", command=lambda: changeColor("blue"), width=2, height=1)
# red2 = tk.Button(colorFrame, bg="#961E5A", command=lambda: changeColor("red2"), width=2, height=1)
# wheat = tk.Button(colorFrame, bg="#F06436", command=lambda: changeColor("wheat"), width=2, height=1)
# crush = tk.Button(colorFrame, bg="#C14B2D", command=lambda: changeColor("crush"), width=2, height=1)
# diva = tk.Button(colorFrame, bg="#E54C68", command=lambda: changeColor("diva"), width=2, height=1)

red.grid(row=0, column=0, padx=2, pady=3)
pink.grid(row=0, column=1, padx=3, pady=3)
blue.grid(row=0, column=2, padx=3, pady=3)
red2.grid(row=0, column=3, padx=3, pady=3)
wheat.grid(row=0, column=4, padx=3, pady=3)
# crush.grid(row=1, column=0, padx=3, pady=3)
# diva.grid(row=1, column=1, padx=3, pady=3)

# imgLabel = Label(root)
# imgLabel.pack()
start.pack()
selectColor.pack()
colorFrame.pack()
colors = {"blue": (113, 43, 114),
          "red": (0, 0, 255),  # B4474A
          "pink": (255, 0, 255),
          "wheat": (54, 100, 240),
          "red2": (90, 30, 150),
          "crush": (76, 70, 141),
          "diva": (104, 76, 229),
          "selected": (0, 0, 255)
          }
lipstick_color = "selected"

# start mainloop
index = 0
root.protocol("WM_DELETE_WINDOW", on_quit)
root.title("VirtualMakeUP")
root.deiconify()
exitFlag = False
pause = False
root.mainloop()
# cv2.destroyAllWindows()
