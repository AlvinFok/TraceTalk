#%%
# from pathlib import Path
# videoFile = Path("./usedVideos/cam1-2022-04-16_14-00-11__43__2.mkv")
# outputDir = "./videoTest_sortOH/"
# print(Path(outputDir) / videoFile.name.replace(videoFile.suffix, "_rewrite"+videoFile.suffix))()
# a = [1]
# del a[0]
# print(a)
#%%


import json
import cv2
import random
from pathlib import Path
# IDInfo = {
#     1:[{"frame":2, "x":3, "y":4, "w":5, "h":6}, {"frame":3, "x":4, "y":5, "w":6, "h":7}]
# }

outputDir = "./videoTest_sortOH/"
videoFile = Path("./usedVideos/cam1-2022-04-16_14-00-11__43__2.mkv")
jsonFile = Path("./videoTest_sortOH/cam1-2022-04-16_14-00-11__43__1_IDInfo.json")
drawID = [129, 133, 135, 151]
IDInfo = None
frameID = 0
color = dict()

print("Reading Json File.")
with open(jsonFile) as d:
    IDInfo = json.load(d)

print("Json File Read.")

print("Reading video")
video = cv2.VideoCapture(str(videoFile))
H = int(video.get(4))
W = int(video.get(3)) 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(Path(outputDir) / videoFile.name.replace(videoFile.suffix, "_rewrite"+videoFile.suffix)), fourcc, 20.0, (W, H))

def main():
    global frameID
    while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        for id in drawID:
            id = str(id)
            if len(IDInfo[id]) >0 and frameID == IDInfo[id][0]["frame"]:#this frame need to draw this id
                cx, cy, w, h = IDInfo[id][0]["x"], IDInfo[id][0]["y"], IDInfo[id][0]["w"], IDInfo[id][0]["h"]
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                idColor = getColor(id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), idColor, 3)
                del IDInfo[id][0]
        
        writer.write(frame)
        frameID += 1
    writer.release()
    video.release()
    print("Done!")
    
def getColor(id):
    global color
    if color.get(id) == None:
        color[id] = (random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255))


if __name__ == "__main__":
    main()