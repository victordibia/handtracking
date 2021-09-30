import cv2

video_source, width, height = 1, 320, 240
cap = cv2.VideoCapture(video_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cv2.namedWindow("Handtrack", cv2.WINDOW_NORMAL)

# ret, image_np = cap.read()
# print(ret, image_np)
def list_ports():
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(
                    "Port %s is working and reads images (%s x %s)" % (dev_port, h, w),
                    img.shape,
                )
                working_ports.append(dev_port)
            else:
                print(
                    "Port %s for camera ( %s x %s) is present but does not reads."
                    % (dev_port, h, w)
                )
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports


# print(list_ports())
while True:
    # Capture frame-by-frame
    ret, image_np = cap.read()
    # Our operations on the frame come here
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    cv2.imshow("Handtrack | Press Q to Exit", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()