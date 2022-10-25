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

outputDir = Path("./videoTest_sortOH/")
videoFile = Path("./usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv")
jsonFile = Path("./videoTest_sortOH/Test_IDInfo.json")
drawID = [108, 129, 133, 135, 146, 151, 192, 195]

IDInfo = None
frameID = 0
color = dict()
skipFrame = 2
saveFrame = True

print("Reading Json File.")
with open(jsonFile) as d:
    IDInfo = json.load(d)

print("Json File Read.")

print("Reading video")
# print(str(outputDir / "frames" / videoFile.name.replace(videoFile.suffix, f"_rewrite_{frameID}.jpg")))

video = cv2.VideoCapture(str(videoFile))
H = int(video.get(4))
W = int(video.get(3)) 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(outputDir /  videoFile.name.replace(videoFile.suffix, "_rewrite"+videoFile.suffix)), fourcc, 20.0, (W, H))

def main():
    global frameID
    while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        
        
        
        if len(IDInfo['133']) > 0 and frameID == IDInfo['133'][0]["frame"] and len(IDInfo['135']) > 0 and frameID == IDInfo['135'][0]["frame"]:
            cx, cy, w, h = IDInfo['135'][0]["x"], IDInfo['135'][0]["y"], IDInfo['135'][0]["w"], IDInfo['135'][0]["h"]
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            idColor = getColor('135')
            cv2.rectangle(frame, (x1, y1), (x2, y2), idColor, 3)
            cv2.putText(frame, '135', (cx, cy - 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0,255,0), thickness=2)
            cv2.imwrite(str(outputDir / "frames" / videoFile.name.replace(videoFile.suffix, f"_rewrite_{frameID}.jpg")), frame)
            
            del IDInfo['133'][0]
            del IDInfo['135'][0]
            
            
        # print(f"{frameID}")
        for id in drawID:
            id = str(id)
            
            if len(IDInfo[id]) > 0 and frameID == IDInfo[id][0]["frame"]:#this frame need to draw this id
                cx, cy, w, h = IDInfo[id][0]["x"], IDInfo[id][0]["y"], IDInfo[id][0]["w"], IDInfo[id][0]["h"]
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                idColor = getColor(id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), idColor, 3)
                cv2.putText(frame, id, (cx, cy - 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0,255,0), thickness=2)
                cv2.imwrite(str(outputDir / "frames" / videoFile.name.replace(videoFile.suffix, f"_rewrite_{frameID}.jpg")), frame)
                
                del IDInfo[id][0]
        frameID += 1
        writer.write(frame)
        
    writer.release()
    video.release()
    print("Done!")
    print("Save as " + str(outputDir /  videoFile.name.replace(videoFile.suffix, "_rewrite"+videoFile.suffix)))
    
def getColor(id):
    global color
    if color.get(id) == None:
        color[id] = (random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255))
        
    return color[id]


if __name__ == "__main__":
    main()