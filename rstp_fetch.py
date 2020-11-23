import cv2
cap = cv2.VideoCapture("rtsp://192.168.142.201/live1.sdp")
# cap = cv2.VideoCapture("rtsp://admin:8848chaffee@192.168.183.63:554/h264/ch1/main/av_stream")
# cap = cv2.VideoCapture("rtsp://r_user:123456@192.168.142.2:554/cam/realmonitor?channel=11&subtype=0")


while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is not None:
        _frame = frame.copy()
        print(_frame.shape)
        _frame = cv2.resize(_frame,dsize=None,fx=0.3,fy=0.3)
        cv2.imshow('frame', _frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()