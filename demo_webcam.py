import argparse
import cv2
import numpy as np
from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal",
                help='Network Type: normal / tiny / prn / v4-tiny')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg",
                "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg",
                "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg",
                "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg",
                "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
    image = frame
else:
    rval = False
pressed_key = cv2.waitKey(1)

far_points = []
isDrawing = False

while rval:
    width, height, inference_time, results = yolo.inference(frame)
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)
        # image = frame
        croppedImage = image.copy()
        xmin = int(x)
        ymin = int(y)
        xmax = int(x+w)
        ymax = int(y+h)

        # cv

        croppedImage = croppedImage[ymin:ymax, xmin:xmax]
        min_YCrCb = np.array([80, 133, 77], np.uint8)
        max_YCrCb = np.array([255, 173, 127], np.uint8)
        imageYCrCb = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2YCR_CB)

        # Find region with skin tone in YCrCb image
        skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        _, contours, _ = cv2.findContours(
            skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw a bounding box rectangle and label on the image
        if contours is not None and len(contours) is not 0:
            contours = max(contours, key=lambda x: cv2.contourArea(x))
            # cv2.drawContours(croppedImage, [contours], -1, (255,255,0), 2)
            maxContour = contours
            hull = cv2.convexHull(contours, returnPoints=False)
            c = contours
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extLeft1 = (extLeft[0]+xmin, extLeft[1]+ymin)
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            extRight1 = (extRight[0]+xmin, extRight[1]+ymin)
            extTop1 = (extTop[0]+xmin, extTop[1]+ymin)
            extBot1 = (extBot[0]+xmin, extBot[1]+ymin)
            cv2.circle(image, extLeft1, 8, (0, 0, 255), -1)
            cv2.circle(image, extRight1, 8, (0, 255, 0), -1)
            cv2.circle(image, extTop1, 8, (255, 0, 0), -1)
            cv2.circle(image, extBot1, 8, (255, 255, 0), -1)
            if len(far_points) > 100:
                far_points.pop(0)

            if pressed_key & 0xFF == ord('d'):
                isDrawing = True
            if pressed_key & 0xFF == ord('c'):
                isDrawing = False
            if pressed_key & 0xFF == ord('x'):
                far_points.clear()
                # cv2.line(canvas, far_points[i], far_points[i+1], (0,0,0), 20)
            if isDrawing:
                far_points.append(extTop1)
                for i in range(len(far_points)-1):
                    cv2.line(image, far_points[i],
                             far_points[i+1], (255, 5, 255), 10)

        # color = (0, 255, 255)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # text = "%s (%s)" % (name, round(confidence, 2))
        # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, color, 2)

    cv2.imshow("preview", image)

    rval, image = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
