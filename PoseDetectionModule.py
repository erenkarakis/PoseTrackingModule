import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, smooth_landmarks=True,
                    detectionCon=0.5, trackCon=0.5, zAxisMultiplier = 1000):
        
        self.mode = mode
        self.smooth_landmarks = smooth_landmarks
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.zAxisMultiplier = zAxisMultiplier

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth_landmarks,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, 
                                                self.mpPose.POSE_CONNECTIONS)
        return img

    def spotLandmarks(self, img, draw=True):
        lmlist = []

        if self.results.pose_landmarks:
            for id, lms in enumerate(self.results.pose_landmarks.landmark):

                h, w, c = img.shape
                cx, cy, cz = int(lms.x*w), int(lms.y*h), int(lms.z*(w/h*self.zAxisMultiplier))
                #print(id, cx, cy, cz)
                lmlist.append([id, cx, cy, cz])

                if draw:
                    cv2.circle(img, (cx, cy), 10, (255,255,0), cv2.FILLED)
        return lmlist
        

def main():
    cap = cv2.VideoCapture("TestVideos/test1.mp4")
    cTime = 0
    pTime = 0

    detector = poseDetector()

    while True:
        success, img = cap.read()

        img = detector.findPose(img, draw=False)
        lmlist = detector.spotLandmarks(img, draw=False)
        if len(lmlist) != 0:
            cv2.circle(img, (lmlist[12][1], lmlist[12][2]), 10, (255, 255, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 
                        3, (255, 0, 255), 3)

        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()