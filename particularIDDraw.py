#%%
import cv2
from pathlib import Path
from libs.utils import *
# IDInfo = {
#     1:[{"frame":2, "x":3, "y":4, "w":5, "h":6}, {"frame":3, "x":4, "y":5, "w":6, "h":7}]
# }
#05-14_09 17 21
#05-12-16 44 50 52 53
#05-12-14 105 106
outputDir = Path("./videoTest_BYTE/")
framesDir = outputDir / "frames/"
framesDir.mkdir(parents=True, exist_ok=True)
videoFile = Path("./videos/FirstRestaurant/cam1-2023-05-12_14-00-22__68__65.mkv")
txtFile = Path("./videos/FirstRestaurant/hardPredict_5/cam1-2023-05-12_14-00-22__68__65.txt")
drawID = [105,106]

IDInfo = None
saveFrame = False
isPadding = True

countInArea_cal = np.array([[216, 750],[216, 350],[580, 200],[663, 0],[1480, 280],[1370, 790], [1080, 850]])
countOutArea = np.array([[184, 990], [184, 283],[518, 130],[569, 144],[663, 0],[1480, 278],[1480, 750],[1280, 1050]])
            
# Define borders (in pixels)
if isPadding:
    top, bottom, left, right = 0, 400, 200, 0
else:
    top, bottom, left, right = 0, 0, 0, 0
print("Reading Txt File.")

IDInfo = getNpData(txtFile)

print("Txt File Read.")


print("Reading video")
# print(str(outputDir / "frames" / videoFile.name.replace(videoFile.suffix, f"_rewrite_{frameID}.jpg")))

video = cv2.VideoCapture(str(videoFile))
H = int(video.get(4))
W = int(video.get(3)) 
FPS = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(str(outputDir /  videoFile.name.replace(videoFile.suffix, "_rewrite"+videoFile.suffix)), fourcc, FPS, (W+left+right, H+top+bottom))

def main():
    frameID = 0
    while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        
        if isPadding:
            frame = padding(frame)
        
        dets = getDets(IDInfo, frameID)
        # print(dets)
        for id in drawID:
            indices = np.where(dets[:, 1] == id)
            det = dets[indices]
        
            if det.size != 0:#this frame need to draw this id
                
                _, _, cx, cy, w, h, _ = det[0]
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                x1, x2 = x1 + left, x2 +left
                cx = cx + left
                idColor = getColor(int(id))
                cv2.rectangle(frame, (x1, y1), (x2, y2), idColor, 3)
                cv2.putText(frame, str(id), (int(cx), int(cy - 7)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0,255,0), thickness=2)
        
        
        # cv2.imwrite(str(outputDir / "frames" / videoFile.name.replace(videoFile.suffix, f"_rewrite_{frameID}.jpg")), frame)
        cv2.polylines(frame, pts=[countInArea_cal], isClosed=True, color=(0,0,255), thickness=3)#draw square area
        cv2.polylines(frame, pts=[countOutArea], isClosed=True, color=(255,0,0), thickness=3)#draw square area
                
        frameID += 1
        writer.write(frame)
        
    writer.release()
    video.release()
    print("Done!")
    print("Save as " + str(outputDir /  videoFile.name.replace(videoFile.suffix, "_rewrite"+videoFile.suffix)))
    
def getColor(id):
    return ( ( id * 10 + 56) % 255, (id * 52 + 3) % 255, (id * 2 + 100) % 255 )

def padding(img):

    # Define border color (in BGR format)
    border_color = [255, 255, 255]

    # Add padding to the image
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)
    return padded_img

if __name__ == "__main__":
    main()
# %%
