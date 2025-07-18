import cv2
import multiprocessing as mp
import time
import numpy as np
import datetime
import imutils

# Detecting algo, found in class to capsulate the algo
class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    # Basic VMD, using supplied code which implement in class
    def detect(self, frame):
        # Convert image from one color space Blue, Green, Red grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        detections = []

        # Deal with first frame 
        if self.prev_frame is None:
            self.prev_frame = gray
            return detections

        # Calculates the absolute difference between two arrays 
        diff = cv2.absdiff(gray, self.prev_frame)
        # Convert a grayscale image into a binary image.
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        # Performing morphological dilation on images
        thresh = cv2.dilate(thresh, None, iterations=2)
        # Find contours in a binary image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Simplify working with contours
        cnts = imutils.grab_contours(cnts)

        # Now i have the number of Contours
        # I want to create a box with proper area around object not for to small objects
        for c in cnts:
            if cv2.contourArea(c) < 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append((x, y, w, h))

        self.prev_frame = gray
        return detections


def streamer(video_path, frame_queue):
    cap = cv2.VideoCapture(video_path) # init the video
    while True:
        ret, frame = cap.read() # read frames (result (ok,fail), frame)
        if not ret:
            break
        frame_queue.put(frame) # place frame in queue
        time.sleep(0.03) # frame produce rate seems ok no lags
    cap.release() #  Closes the video file or capturing device and releases associated resources
    
    frame_queue.put(None) # Part 3 End of process send to detector to end


def detector(frame_queue, detect_queue):
    motion_detector = MotionDetector() # need to init prev_fram (As detect algo)

    # Wait until frame receive
    while True:
        frame = frame_queue.get() 
        # Part 3 stop process  
        if frame is None:
            break

        detections = motion_detector.detect(frame) # detect and retrun all motion detection 
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # add timestamp
        detect_queue.put((frame, detections, timestamp)) # tuple, to present process

    detect_queue.put(None) # Part 3 End of process send to presenter to end


def presenter(detect_queue):
    # Wait to receive frame,detections, and time 
    while True:
        data = detect_queue.get()
        # Part 3 close process 
        if data is None:
            break

        frame, detections, timestamp = data # parse to individual variable
        
        # Part 2 add blurring 
        for (x, y, w, h) in detections:
            # Region of interest 
            roi = frame[y:y+h, x:x+w]
            # Tried 3x3 7x7 and 15x15 seems that 15x15 the blurriest 
            blurred = cv2.GaussianBlur(roi, (15, 15), 0)
            frame[y:y+h, x:x+w] = blurred
    
            # cv2.rectangle
            # The image to draw on
            # start point, Top-left corner of the rectangle
            # end point, Bottom-right corner of the rectangle
            # Color of the rectangle (green in BGR)
            # Line thickness
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Place timestamp in frame 
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame in a window 
        cv2.imshow("surveillance camera", frame)
        
        # Render it on the screen 10ms
        cv2.waitKey(10)

    # Close all windows in end of videos 
    cv2.destroyAllWindows()


def main():

    video_path = "People - 6387.mp4" # the video input 

    # Consumer Producer style, one process feeds the follow
    # 2 queues one hold frames(streamer->detector), second holds frame and detection(detector->present) 
    # Help if one process slower than other it queue frames 
    frame_queue = mp.Queue(maxsize=10)
    detect_queue = mp.Queue(maxsize=10)

    streamer_process = mp.Process(target=streamer, args=(video_path, frame_queue))
    detector_process = mp.Process(target=detector, args=(frame_queue, detect_queue))
    presenter_process = mp.Process(target=presenter, args=(detect_queue,))

    streamer_process.start()
    detector_process.start()
    presenter_process.start()

    # Part 3 - in end video each queue in order send None to stop process 
    streamer_process.join()
    detector_process.join()
    presenter_process.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()