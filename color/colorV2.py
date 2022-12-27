import torch
import cv2
import numpy as np
img_path="cars7.png"

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
color_model = torch.load('color.pb')
color_model.to(device)
color_model.eval()

image = cv2.imread(img_path)

img = cv2.resize(image, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.reshape(1, 224, 224, 3) / 255

predictions = color_model(torch.Tensor(img).to(device)).detach().cpu().numpy()
availabe_colors=sorted(['black','blue','brown','green','pink','red','silver','white','yellow'])

print(availabe_colors[np.argmax(predictions)],": ",predictions[0][np.argmax(predictions)]*100,"%")
