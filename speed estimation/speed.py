import cv2
import json
import torch

# how this work
# We locate two points which we know the distance between them  in real world in meters (street)
# If the car passed at the first point we start counting frames until car reach to endpoint
# and after that we can get the speed by divide distance in realworld over time. we can get time by multiply fps * frame count


def init(video_path="00.mp4"):
    video = cv2.VideoCapture(video_path)
    suc, image = video.read()

    cv2.namedWindow("select the area", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("select the area", image)
    start_point = (int(roi[0]), int(roi[1]))
    end_point = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
    print("start point")
    print(start_point)
    print("end point")
    print(end_point)
    import io
    with io.open('data.json', 'w', encoding='utf8') as outfile:
        # distance_threshold = 50 pixel
        str_ = json.dumps(
            {"start_point": start_point, "end_point": end_point, "distance": 1.5, "distance_threshold": 50},
            indent=4,
            sort_keys=False,
            separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)

# init()
# exit()


with open('data.json') as data_file:
    data = json.load(data_file)


class CarSpeed:
    def __init__(self, fps):
        self.frame_count = 0
        self.fps = fps
        self.before_area = True
        self.in_area = False
        self.after_area = False

    def get_state(self, pos):
        #     return 0 to do nothing
        #     return 1 to increase frame count
        #     return 2 to calculate_speed
        if self.before_area:
            distance =abs(pos[0] - data['start_point'][0])
            if distance < data['distance_threshold']:
                self.before_area=False
                self.in_area=True
                return 1
            else:
                return 0

        if self.in_area:
            distance =abs(pos[0] - data['end_point'][0])
            if distance < data['distance_threshold']:
                self.in_area = False
                self.after_area = True
                return 2
            else:
                return 1

        if self.after_area:
            return 2

    def increase_frame_count(self):
        self.frame_count += 1
        print("frame count")
        print(self.frame_count)

    def calculate_speed(self):
        # meter per second
        t = self.frame_count / self.fps
        return data['distance'] / t


cap = cv2.VideoCapture("00.mp4")

print(cap.get(cv2.CAP_PROP_FPS))

car1 = CarSpeed(fps=cap.get(cv2.CAP_PROP_FPS))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../cars_detection/car_detection.pt')
model.to(device)
model.eval()

flag=False
flag2=False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    detection_boxes = model(frame)
    for indx, bbox in enumerate(detection_boxes.xywh[0]):
        x = int(bbox[0].item())
        y = int(bbox[1].item())
        w = int(bbox[2].item())
        h = int(bbox[3].item())

        x = int(x - w / 2)
        y = int(y - h / 2)

        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)

        r=car1.get_state((cx,cy))

        if r==0:
            pass
        elif r==1:
            if not flag:
                cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                cv2.circle(frame, (data['start_point'][0],data['start_point'][0]), 7, (0, 255, 255), -1)

                cv2.imwrite("ToArea.jpg",frame)
                flag=True

            car1.increase_frame_count()
        elif r==2:
            if not flag2:
                cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                cv2.circle(frame, (data['end_point'][0], data['end_point'][0]), 7, (0, 255, 255), -1)

                cv2.imwrite("OutArea.jpg", frame)
                flag2 = True
            print("speed")
            print("%.2f" %car1.calculate_speed(),"m/s")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# calculate time from start point to end point

# initialize speed
#       1- add start point
#       2- add end point
#       3- add distance

# measure the distance between start point and car position


# if the result bigger than threshold then it will ignore
# else we will start calculate number of frames that car appears for this car until go to end point
