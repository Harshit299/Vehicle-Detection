import cv2 as cv
import numpy as np
import time

# initializing video capture
capture = cv.VideoCapture("D:\\OpenCV\\projects\\video.mp4")

# video writer to save output video
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 30.0, (int(capture.get(3)), int(capture.get(4))))

min_rect_width = 80 
min_rect_height = 80  

count_line_pos = 550  # line position for counting

# defining lane boundaries (x-coordinates)
lanes = [(0, 600), (600, 1200)]
lane_counters = [0, 0]

# initializing background subtractor
bg_subt = cv.bgsegm.createBackgroundSubtractorMOG()

def center_pt_calc(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

detect = []
offset = 6  # permissible error for the detection line
total_counter = 0 

presentTime = 0
while True:
    try:
        success, img = capture.read()

        if not success:
            break

    except Exception as e:
        print(f"Error reading frame: {e}")
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 5)

    # applying background subtractor
    img_sub = bg_subt.apply(blur)
    dilate = cv.dilate(img_sub, np.ones((5, 5)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilatada = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)
    dilatada = cv.morphologyEx(dilatada, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(dilatada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.line(img, (0, count_line_pos), (1200, count_line_pos), (0, 255, 0), 3)

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(c)
        validate_counter = (w >= min_rect_width) and (h >= min_rect_height)
        if not validate_counter:
            continue

        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        center = center_pt_calc(x, y, w, h)
        detect.append(center)

        for (cx, cy) in detect[:]:
            if count_line_pos - offset < cy < count_line_pos + offset:
                for idx, (lane_start, lane_end) in enumerate(lanes):
                    if lane_start < cx < lane_end:
                        lane_counters[idx] += 1
                        total_counter += 1  
                        detect.remove((cx, cy))
                        print(f"Lane {idx + 1} Vehicle Counter: {lane_counters[idx]}")
                        print(f"Total Vehicle Counter: {total_counter}")

    # display vehicle count for each lane and total vehicle count on the frame
    for idx, (lane_start, lane_end) in enumerate(lanes):
        cv.putText(img, f"Lane {idx + 1} Counter: {lane_counters[idx]}", (lane_start + 10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
    
    cv.putText(img, f"Total Vehicles: {total_counter}", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    # write the frame to the output video
    out.write(img)

    # calculating fps
    currentTime = time.time()
    if presentTime == 0:
        fps = 0
    else:
        fps = 1 / (currentTime-presentTime + 1e-6)

    presentTime=currentTime

    # display fps on video frame
    cv.putText(img, f"FPS:{str(int(fps))}", (630,180), cv.FONT_HERSHEY_DUPLEX, 3, (0,255,0), 3)

    # display video frame
    cv.imshow("Vehicle Detector", img)

    if cv.waitKey(35) & 0xFF == ord('k'):
        break

capture.release()
out.release()
cv.destroyAllWindows()