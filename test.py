#%%
import json
with open("evaluation_BYTE.json", 'a+') as outFile:
    outFile.seek(0)
    data = json.load(outFile)
    outFile.truncate(0)
    print(data)
    print("args.yolo_thresh=0.4,args.track_thresh=0.6,args.track_buffer=30,args.match_thresh=0.9" in data)
    json.dump({1:2}, outFile)
    
#%%
from shapely.geometry import Point, Polygon
import numpy as np
countInArea_cal = np.array([[0, 1100],[0, 100],[557, 100],[983, 260], [993, 359],[1159, 493],[1137, 586],[1100, 590],[1425, 1007],[1525, 985],[1574, 814],[1930, 1100] ])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
countOutArea = np.array([[0, 1080],[0, 0],[877, 0],[1019, 257],[1007, 360],[1177, 501],[1165, 595],[1512, 962],[1609, 578], [1980, 728], [1980, 1080]])
countInAreaPolygon = Polygon(countInArea_cal)
countOutAreaPolygon = Polygon(countOutArea)

point = Point((1361, 1091))
point2 = Point((1370,1066))

print(point.within(countInAreaPolygon))
print(point.within(countOutAreaPolygon))
print()
print(point2.within(countInAreaPolygon))
print(point2.within(countOutAreaPolygon))

# %%
import matplotlib.pyplot as plt
IN = [92.17, 100, 98.62]
OUT = [94.90, 97.45, 98.98]
x = [5, 12, 25]
plt.plot(x, IN, label="IN_Accuracy", marker='.')
plt.plot(x, OUT, label="OUT_Accuracy", marker='.')
plt.legend()
plt.xlabel("FPS")
plt.ylabel("Accuracy")
plt.title("Performance with different FPS")
plt.xticks(x)
plt.show()

# %%
