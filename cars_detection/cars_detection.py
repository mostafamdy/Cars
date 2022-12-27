import cv2
import torch


img_path="cars1.jpg"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('ultralytics/yolov5', 'custom',path='car_detection.pt')
model.to(device)
model.eval()

img=cv2.imread(img_path)
bboxes=model(img)
detection_th = .0
for indx, bbox in enumerate(bboxes.xywh[0]):
    if bboxes.xywh[0][indx][4] < detection_th:
        continue
    x = int(bbox[0].item())
    y = int(bbox[1].item())
    w = int(bbox[2].item())
    h = int(bbox[3].item())
    x = int(x - w / 2)
    y = int(y - h / 2)
    cx = int((x + x + w) / 2)
    cy = int((y + y + h) / 2)

    cv2.imwrite('result cropped_car '+str(indx)+'.jpg', img[y:y + h, x:x + w])
    img=cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite('result.jpg', img)