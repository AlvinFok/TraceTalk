# People-Flow-System
六燃人流系統
# install tutorial


## Step 1. 下載檔案
```bash=
git clone https://github.com/AlvinFok/People-Flow-System-of-the-Sixth-Japanese-Navy-Fuel-Plant

cd YoloTalk
git clone https://github.com/AlexeyAB/darknet.git
git clone https://github.com/ifzhang/ByteTrack.git
```

## Step 2. 編譯 darknet
進入```darknet```資料夾，並根據本身電腦環境，編輯```Makefile```檔案

參考:
```bash=
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=0
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```

接著進行編譯
```bash=
make
```




## Step 3. 安裝套件並執行範例程式
> 可先自行安裝python虛擬環境

> 依ByteTrack 指示安裝[ByteTrack](https://github.com/ifzhang/ByteTrack.git)

> yolox 版本使用```bytetrack_x_mot17.pth.tar```

```bash=
pip install -r requirements.txt
python3 test_BYTE_YOLOX.py --video 0325__12__12.mp4 -f ByteTrack/exps/example/mot/yolox_x_mix_det.py
 -c ByteTrack/pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse
```

執行完成後，會自動產生demo資料夾，儲存辨識結果


## Step 4. (Option) 設置 YoloDevice 物件參數
> YoloTalk 提供許多參數可供設置。
> 修改YoloDevice物件內的參數，以符合自己的需求。

編輯 ```main.py```

- ```config_file``` (string)： config 檔案路徑
- ```data_file``` (string)： data 檔案路徑
- ```weights_file``` (string)： weight 檔案路徑
- ```output_dir ``` (string)： 輸出結果的路徑 (若路徑不存在，則會自動建立)
- ```thresh``` (int): 辨識靈敏度 (範圍 0~100)
- ```video_url``` (string)： 欲辨識的影像 (影片路徑或串流影像網址)
- ```is_threading``` (bool)： 若為影片，則設為 False; 若為即時串流影像，則設為 True
- ```vertex``` (list or None)： 若欲辨識特定範圍，則輸入辨識位置，如 [(0,0),(100,0),(100,100),(0,100)]。若欲辨識全部範圍，則設為 None
- ```target_classes```(list or None)： 若欲辨識特定的class名稱，則設置 如 ["person"]。若欲辨識全部class，則設為 None (設為None，開啟obj_trace暫時有bug)
- ```alias``` (string)：設置此影片的別名 (為辨識結果資料夾及影片命名)
- ```display_message``` (bool)： 若欲於 Terminal 顯示相關資訊，則設為 True，否則設為 False
- ```obj_trace``` (bool)： 若欲開啟物件追蹤，則設為 True，並於 on_data() 回傳 id 名稱。否則為 False。
- ```save_img``` (bool)： 若欲將辨識結果存成照片，則設 True。否則設 False
- ```save_video``` (bool)： 若欲將辨識結果存成影像，則設為True。否則設為 False
- ```auto_restart``` (bool): 若當無法讀取串流影像時，想自動重新啟動程式，則設為True。否則設為False。

## Step 5: (Option) 設置IoTtalk與LineBot
> 設置IoTTalk與LineBot的連結

### IoTtalk setting

#### IDF
One IDF includes 4 variables:
- **object_id:** Object name.
- **coordinate_x:** x coordinate of the detected object.
- **coordinate_y:** y coordinate of the detected object.
- **pointer to yolo device DB:** when detecting object, yolo device will save the frame in the path that user set.

Take yPerson-I as an example:
![](https://i.imgur.com/MpVL2iw.png =70%x)

#### Notice
- The object type should be set according to the image above.

#### Model
Chose the IDF you need, you can detect more than one object ( take yPerson-I for example ).

![](https://i.imgur.com/3Px2k1N.png)

#### GUI connection
IoTtalk GUI connection example:

![](https://i.imgur.com/ot3qVwn.png =70%x)


### LINE Notify

1. Use your LINE account to sign in LINE [Notify](https://notify-bot.line.me/zh_TW/).
點選 Generate token

2. Set token name and chose the chat group which will get message from LINE Notify.
3. Make sure to remember the access token.
4. `vim LineNotify.py`
past the token in `token_key = ''`


![](https://i.imgur.com/m7UcJnS.png)

#### 修改 ```main.py``` 程式碼
>修改以下對應參數

```python=
ServerURL = 'https://edgecore.iottalk.tw' # set the url to your own iottalk server   
Reg_addr = '555642434' # if None, Reg_addr = MAC address
DAN.profile['dm_name']='Yolo_Device'
DAN.profile['df_list']=['yPerson-I',]
DAN.profile['d_name']= 'YOLOjim'

DAN.device_registration_with_retry(ServerURL, Reg_addr)
#DAN.deregister()  # if you want to deregister this device, uncomment this line
#exit()            # if you want to deregister this device, uncomment this line
```

--- 

## :bulb: TroubleShooting
若編譯darknet時遇到以下錯誤，可能是cuda版本與新版本daeknet不相容

```bash=
./src/network_kernels.cu(721): error: identifier "cudaStreamCaptureModeGlobal" is undefined

./src/network_kernels.cu(721): error: too many arguments in function call

2 errors detected in the compilation of "/tmp/tmpxft_0019d7fa_00000000-10_network_kernels.compute_70.cpp1.ii".
Makefile:185: recipe for target 'obj/network_kernels.o' failed
make: *** [obj/network_kernels.o] Error 1
```

編輯```src/network_kernels.cu```，註解```CHECK_CUDA(cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal));```(約在721行)，並執行以下指令，重新編譯 darknet

```bash=
make clean
make
```

# Format of testing videos
```
${dateTime}__${getIn}__${current}.mkv
```