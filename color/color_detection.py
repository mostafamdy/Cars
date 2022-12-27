import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

img_path="sky.jpg"

index=["color_name","hex","R","G","B"]

csv = pd.read_csv('colors.csv', names=index, header=None)


def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        # print(i)
        # print(csv.loc[i,"R"])
        # print(csv.loc[i, "G"])
        # print(csv.loc[i, "B"])
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    # print("cname")
    # print(cname)
    return cname


# to test getColorName
# print(getColorName(239,153,182))
# exit()

dominat_colors=[]
def get_color(car_img):
    img = car_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_img = np.reshape(img, (-1, 3))
    kmeans = KMeans(n_clusters=5, random_state=101)
    kmeans.fit(flat_img)
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    print("top 5 colors")
    print(dominant_colors)
    globals()['dominat_colors']=dominant_colors
    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
    print("percentages")
    print(percentages)
    (R,G,B)=dominant_colors[np.argmax(percentages)]

    print(R,G,B)
    return [getColorName(R,G,B),(int(R),int(G),int(B))]

img=cv2.imread(img_path)
color=get_color(img)
print(color)

# testing
from PIL import Image
import numpy as np

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[:,:int(w*1/5)]= dominat_colors[0]
data[:,int(w*1/5):int(w*2/5)]= dominat_colors[1]
data[:,int(w*2/5):int(w*3/5)]= dominat_colors[2]
data[:,int(w*3/5):int(w*4/5)]= dominat_colors[3]
data[:,int(w*4/5):int(w*5/5)]= dominat_colors[4]
# ".join(dominat_colors[4])
print(type(dominat_colors[0]))
data=cv2.putText(data,"("+str(dominat_colors[0][0])+","+str(dominat_colors[0][1])+","+str(dominat_colors[0][2])+")",(0,50), cv2.FONT_HERSHEY_SIMPLEX,.5, (0,0,0), 1, cv2.LINE_AA)
data=cv2.putText(data,"("+str(dominat_colors[1][0])+","+str(dominat_colors[1][1])+","+str(dominat_colors[1][2])+")",(int(w*1/5),75), cv2.FONT_HERSHEY_SIMPLEX,.5, (0,0,0), 1, cv2.LINE_AA)
data=cv2.putText(data,"("+str(dominat_colors[2][0])+","+str(dominat_colors[2][1])+","+str(dominat_colors[2][2])+")",(int(w*2/5),100), cv2.FONT_HERSHEY_SIMPLEX,.5, (0,0,0), 1, cv2.LINE_AA)
data=cv2.putText(data,"("+str(dominat_colors[3][0])+","+str(dominat_colors[3][1])+","+str(dominat_colors[3][2])+")",(int(w*3/5),125), cv2.FONT_HERSHEY_SIMPLEX,.5, (0,0,0), 1, cv2.LINE_AA)
data=cv2.putText(data,"("+str(dominat_colors[4][0])+","+str(dominat_colors[4][1])+","+str(dominat_colors[4][2])+")",(int(w*4/5),150), cv2.FONT_HERSHEY_SIMPLEX,.5, (0,0,0), 1, cv2.LINE_AA)
img = Image.fromarray(data, 'RGB')
img.save('top5Colors.png')

data = np.zeros((h, w, 3), dtype=np.uint8)
data[:,:]= color[1]
img = Image.fromarray(data, 'RGB')
img.save('my.png')