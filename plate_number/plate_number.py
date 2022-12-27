import torch
import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
img_path = "usa3.jpg"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path="plate_number.pt")
plate_model.to(device)
plate_model.eval()

img=cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
bboxes=plate_model(img)
img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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

    cv2.imwrite('cropped_licence_plate.jpg',img[y:y + h, x:x + w])
    ocr_result=ocr.ocr(img[y:y + h, x:x + w])

    print("ocr_result")
    for i in ocr_result:
        ocr_text=i[1][0]
        print(ocr_text,str(100*i[1][1])+"%")

    img=cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    r=cv2.imwrite('licence_plate.jpg', img)
    if r:
        print("plate number saved in cropped_licence_plate.jpg")
        print("you can see plate number in original image licence_plate.jpg")
    else:
        print("can't save img")



