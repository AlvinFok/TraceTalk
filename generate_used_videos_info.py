import cv2
import glob
from tqdm import tqdm

OUTPUT_FILENAME = "usedVideos_info.csv"

mkvs = glob.glob("./usedVideos/*.mkv")

# f"{Name} {FPS} {Resolution} {Length} {In} {Out}"
# with open(OUTPUT_FILENAME, "w", encoding="utf-8") as file:
#     file.write("Name,FPS,Resolution,Length,In,Out\n")
#     for mkv in tqdm(mkvs):
#         FILENAME, IN, OUT = mkv.strip(".mkv").split("__")
#         FILENAME = FILENAME.split('/')[-1]
#         OUT = int(IN) - int(OUT)

#         cap = cv2.VideoCapture(mkv)
        
#         count = 0
#         while cap.isOpened():
#             ret, _ = cap.read()
#             if not ret:
#                 break
#             count += 1
        
#         FPS = 25  # int(cap.get(cv2.CAP_PROP_FPS))
#         width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#         height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         RESOLUTION = f"{int(width)}x{int(height)}"
#         # LENGTH = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # TIME = f"{LENGTH / FPS:02.2f}"
        
#         cap.release()
        
#         file.write(f"{FILENAME},{FPS},{RESOLUTION},{count},{IN},{OUT}\n")

duration_check = {
    "cam1-2022-04-16_15-00-14":53903,
    "cam1-2022-05-08_14-00-40":53902,
    "cam1-2022-04-03_14-00-19":53975,
    "cam1-2022-04-13_15-00-26":24750,
    "cam1-2022-04-02_15-00-07":53902,
    "cam1-2022-05-08_16-00-46":53902,
    "cam1-2022-03-25_13-00-43":53900,
    "cam1-2022-04-03_16-00-24":53901,
    "cam1-2022-04-16_14-00-11":53902,
    "cam1-2022-04-03_15-00-21":53902,
    "cam1-2022-05-07_14-00-27":53979,
    "cam1-2022-05-07_17-00-36":26566,
    "cam1-2022-04-16_16-00-17":53902,
    "cam1-2022-04-10_15-00-46":53899,
    "cam1-2022-04-10_14-00-44":53899,
    "cam1-2022-05-07_16-00-33":53900,
    "cam1-2022-05-08_15-00-43":53902,
}

length_total = 0
in_total = 0
out_total = 0

for mkv in tqdm(mkvs):
    try:
        FILENAME, IN, OUT = mkv.strip(".mkv").split("__")
        FILENAME = FILENAME.split('/')[-1]
        duration = duration_check[FILENAME]
        IN = int(IN)
        OUT = IN - int(OUT)    
    except:
        FILENAME = "cam1-2023-01-15_13-00-41"
        IN = OUT = 800
        duration = 53900
    
    length_total += duration
    in_total += IN
    out_total += OUT
    
print(f"{length_total = } {in_total = } {out_total = }")
