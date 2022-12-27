import cv2
import numpy as np
import torch

img_path="cars6.png"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("num_of_gpus ", torch.cuda.device_count())
print("working on  ", device)

logo_detection=torch.hub.load('ultralytics/yolov5', 'custom',path='logo_detection.pt')

detection_result=logo_detection(img_path)
image = cv2.imread(img_path)
croped_image = None

car_logo = detection_result.xywh[0]
print(car_logo)

try:
    x = car_logo[0][0].item()
    y = car_logo[0][1].item()
    w = car_logo[0][2].item()
    h = car_logo[0][3].item()
    prediction = car_logo[0][4].item()

except:
     print("can't detect logo")
     exit()

croped_image = image[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

cv2.imwrite("crop.jpg",croped_image)
# exit()
make_model = torch.load('model.pb')
make_model.to(device)
make_model.eval()

img = cv2.resize(croped_image, (32, 32))
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
img = img.reshape(1, 32, 32, 3) / 255

predictions = make_model(torch.Tensor(img).to(device)).detach().cpu().numpy()
availabe_make=[
    "acura",
    "audi",
    "bmw",
    "cadillac",
    "chevrolet",
    "daewoo",
    "fiat",
    "ford",
    "gmc",
    "honda",
    "hyundai",
    "infiniti",
    "isuzu",
    "jeep",
    "kia",
    "landrover",
    "mazda",
    "mercury",
    "mini",
    "mitsubishi",
    "nissan",
    "porsche",
    "subaru",
    "suzuki",
    "toyota",
    "volkswagen"
    ]

print(availabe_make[np.argmax(predictions)],": ",predictions[0][np.argmax(predictions)]*100,"%")
