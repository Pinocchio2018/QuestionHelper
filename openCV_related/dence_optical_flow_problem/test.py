import numpy as np
import cv2 as cv


def put_frame_no(image, frame_no):
    # font
    font = cv.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 450)

    # fontScale
    font_scale = 2

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    image = cv.putText(image, "frame no: " + str(frame_no), org, font,
                       font_scale, color, thickness, cv.LINE_AA)

    return image


cap = cv.VideoCapture(cv.samples.findFile("0116-sample4-edited-short-throw.mp4"))
ret, frame1 = cap.read()
prv_frame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

cv.namedWindow("flow image", cv.WINDOW_NORMAL)
cv.resizeWindow("flow image", 800, 600)


frame_no = 0
while 1:
    ret, origin_img = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    next_frame = cv.cvtColor(origin_img, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prv_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    flow_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    flow_image = put_frame_no(flow_image, frame_no)
    origin_img = put_frame_no(origin_img, frame_no)

    frame_no += 1

    vis_frame = np.concatenate((origin_img, flow_image), axis=1)

    cv.imshow('flow image', vis_frame)
    # cv.imshow('origin', flow_image)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', origin_img)
        cv.imwrite('opticalhsv.png', flow_image)
    prv_frame = next_frame

cv.destroyAllWindows()
