import json
with open("evaluation_BYTE.json", 'a+') as outFile:
    outFile.seek(0)
    data = json.load(outFile)
    outFile.truncate(0)
    print(data)
    print("args.yolo_thresh=0.4,args.track_thresh=0.6,args.track_buffer=30,args.match_thresh=0.9" in data)
    json.dump({1:2}, outFile)