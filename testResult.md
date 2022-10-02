sortOD edited 15/179
thresh = 0.8 sortOH:age=30 conf_trgt = 0.35, conf_objt = 0.75
{'0325__12.mp4': {'GT': 12, 'predict': 11}, 
'cam1-2022-03-25_13-00-43__27.mkv': {'GT':27, 'predict': 26}, 
'cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 51}
'cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 32}, 
'cam1-2022-04-16_16-00-17__6.mkv': {'GT': 6, 'predict': 7}, 
'cam1-2022-05-08_15-00-43__9.mkv': {'GT': 9, 'predict': 10}}

sort_OH edied 17/179
thresh = 0.5 sortOH:age=30 conf_trgt = 0.35, conf_objt = 0.75 
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 38}}  
{'cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 53}} 
0325 pass
{cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 2}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-05-08_15-00-43__9.mkv': {'GT': 9, 'predict': 10}}



deepsort
5/136(0.036) 8/179(0.044)
thresh = 0.5 DeepSort:age=30 max_cosine_distance = 0.8, nn_budget = None
{'usedVideos/0325__12.mp4': {'GT': 12, 'predict': 13}}
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 5}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 40}}
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 2}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-03-25_13-00-43__27.mkv': {'GT': 27, 'predict': 29}}
usedVideos/cam1-2022-04-16_15-00-14__35.mkv 35 35
usedVideos/cam1-2022-05-08_16-00-46__7.mkv 7 7
cam1-2022-05-07_14-00-27__11 11 11
cam1-2022-05-07_16-00-33__7 7 7
cam1-2022-05-08_15-00-43__9 9 9
cam1-2022-04-02_15-00-07__3 3 3



centroid
28/136 125/179
thresh = 0.5 DeepSort:max_lost=30 centroid_distance_threshold = 50
{'usedVideos/0325__12.mp4': {'GT': 12, 'predict': 14}}
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 10}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 10}}
{'usedVideos/cam1-2022-04-16_16-00-17__6.mkv': {'GT': 6, 'predict': 8}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 47}}28
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 12}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 140}}
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_15-00-43__9.mkv': {'GT': 9, 'predict': 11}}
{'usedVideos/cam1-2022-04-02_15-00-07__3.mkv': {'GT': 3, 'predict': 2}}
{'usedVideos/cam1-2022-03-25_13-00-43__27.mkv': {'GT': 27, 'predict': 29}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 8}}


sortOH original 5/136  22/179
thresh = 0.5 sortOH:age=30 conf_trgt = 0.35, conf_objt = 0.75
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 60}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 40}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-05-08_15-00-43__9.mkv': {'GT': 9, 'predict': 10}}
0325__12 12 12
cam1-2022-05-08_16-00-46__7 7
cam1-2022-05-07_14-00-27__11 11 11
cam1-2022-05-07_17-00-36__3 3 3
cam1-2022-05-07_16-00-33__7 7 7
cam1-2022-03-25_13-00-43__27 27
cam1-2022-04-02_15-00-07__3 3 3

deepsort 
thresh = 0.5 DeepSort:age=15 max_cosine_distance = 0.8, nn_budget = None
{'usedVideos/0325__12.mp4': {'GT': 12, 'predict': 17}}
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 38}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 59}}
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 11}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-03-25_13-00-43__27.mkv': {'GT': 27, 'predict': 30}}
{'usedVideos/cam1-2022-04-02_15-00-07__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_15-00-43__10.mkv': {'GT': 10, 'predict': 10}}

deepsort
thresh = 0.5 DeepSort:age=15 max_cosine_distance = 0.6, nn_budget = None, n_init=10
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 31}}
{'usedVideos/cam1-2022-03-25_13-00-43__28.mkv': {'GT': 28, 'predict': 26}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 51}}
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 11}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-04-02_15-00-07__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_15-00-43__10.mkv': {'GT': 10, 'predict': 10}}


sortOH
thresh = 0.65 sortOH:age=30 conf_trgt = 0.35, conf_objt = 0.75
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 2}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 37}}
{'usedVideos/cam1-2022-03-25_13-00-43__28.mkv': {'GT': 28, 'predict': 26}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 63}}


sortOH no suspiciousAreaHandling
thresh = 0.65 sortOH:age=30 conf_trgt = 0.35, conf_objt = 0.75
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 35}}
{'usedVideos/cam1-2022-03-25_13-00-43__28.mkv': {'GT': 28, 'predict': 25}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 28}}
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 2}}
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 9}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-04-02_15-00-07__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_15-00-43__10.mkv': {'GT': 10, 'predict': 10}}


sortIH no id switch handling  51/147
thresh = 0.65 sortOH:age=30 conf_trgt = 0.35, conf_objt = 0.75
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 42}}
{'usedVideos/cam1-2022-03-25_13-00-43__28.mkv': {'GT': 28, 'predict': 26}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 81}} error:38
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 11}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 8}}





sortOH with ID switch handling 3/179

thresh = 0.5 sortOH:age=30 conf_trgt = 0.35, conf_objt = 0.75
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 35}}
{'usedVideos/cam1-2022-03-25_13-00-43__28.mkv': {'GT': 27, 'predict': 27}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 46}}
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 11}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-04-02_15-00-07__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_15-00-43__10.mkv': {'GT': 10, 'predict': 10}}



centroid with id switch handling
19/160
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 1}}
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-05-08_15-00-43__10.mkv': {'GT': 10, 'predict': 8}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 13}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 9}}
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-04-02_15-00-07__3.mkv': {'GT': 3, 'predict': 4}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 34}}
{'usedVideos/cam1-2022-03-25_13-00-43__27.mkv': {'GT': 27, 'predict': 31}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 40}}


deepsort with ID switch handling
9/160
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-05-08_15-00-43__10.mkv': {'GT': 10, 'predict': 9}}
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 11}}
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 4}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 36}}
{'usedVideos/cam1-2022-03-25_13-00-43__27.mkv': {'GT': 27, 'predict': 27}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-04-02_15-00-07__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 42}}

{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 34}}
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 6}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 46}}
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 11}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-03-25_13-00-43__27.mkv': {'GT': 27, 'predict': 27}}

sortOH with yolo final weight
{'usedVideos/cam1-2022-05-07_17-00-36__3.mkv': {'GT': 3, 'predict': 2}}
{'usedVideos/cam1-2022-05-07_14-00-27__11.mkv': {'GT': 11, 'predict': 12}}
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 37}}
{'usedVideos/cam1-2022-04-02_15-00-07__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_14-00-40__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-05-07_16-00-33__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 10}}
{'usedVideos/cam1-2022-05-08_15-00-43__10.mkv': {'GT': 10, 'predict': 11}}
{'usedVideos/cam1-2022-03-25_13-00-43__29.mkv': {'GT': 29, 'predict': 29}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 64}}

sortOH with yolo 230000 weight
{'usedVideos/cam1-2022-05-08_16-00-46__7.mkv': {'GT': 7, 'predict': 8}}
{'usedVideos/cam1-2022-04-16_16-00-17__7.mkv': {'GT': 7, 'predict': 7}}
{'usedVideos/cam1-2022-04-16_15-00-14__35.mkv': {'GT': 35, 'predict': 41}}
{'usedVideos/cam1-2022-03-25_13-00-43__29.mkv': {'GT': 29, 'predict': 29}}
{'usedVideos/cam1-2022-04-16_14-00-11__43.mkv': {'GT': 43, 'predict': 61}}
{'usedVideos/cam1-2022-04-02_15-00-07__3.mkv': {'GT': 3, 'predict': 3}}
{'usedVideos/cam1-2022-05-08_15-00-43__10.mkv': {'GT': 10, 'predict': 11}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT': 2, 'predict': 7}}


sortOH with new video
{'video/cam1-2022-04-13_15-00-26__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 4, 'GT_Current': 1, 'Pred_Current': 3}}
{'video/cam1-2022-04-10_15-00-46__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': -2}}
{'video/cam1-2022-04-03_16-00-24__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 8, 'GT_Current': 1, 'Pred_Current': 4}}
{'video/cam1-2022-04-10_14-00-44__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 9, 'GT_Current': 1, 'Pred_Current': -6}}
{'video/cam1-2022-04-03_14-00-19__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 1}}
{'video/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 30, 'GT_Current': 1, 'Pred_Current': 3}}


{'video/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 4, 'GT_Current': 1, 'Pred_Current': 3}}
{'video/cam1-2022-04-10_14-00-44__10__-4.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 8, 'GT_Current': -4, 'Pred_Current': -7}}
{'video/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 9, 'GT_Current': -2, 'Pred_Current': -3}}
{'video/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'video/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 1}}
{'video/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': 1}}



sortOH with yolov7 training data
1/105 
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_15-00-43__11__1.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': -5, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 25, 'GT_Current': 1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-05-08_16-00-46__8__0.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 7, 'GT_Current': 0, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-4.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 8, 'GT_Current': -4, 'Pred_Current': -8}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
3/
{'usedVideos/cam1-2022-03-25_13-00-43__29__16.mkv': {'GT_TotalIn': 29, 'Pre__TotalIn': 26, 'GT_Current': 16, 'Pred_Current': 14}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__1.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 34, 'GT_Current': 1, 'Pred_Current': 3}}
{'usedVideos/cam1-2022-04-16_14-00-11__43__1.mkv': {'GT_TotalIn': 43, 'Pre__TotalIn': 50, 'GT_Current': 1, 'Pred_Current': 8}}

0/105 3/102
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 6, 'Pre__TotalIn': 6, 'GT_Current': -6, 'Pred_Current': -6}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 6, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-05-08_16-00-46__8__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 0, 'Pred_Current': -1}}

3/111 2/90
{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 35, 'GT_Current': 4, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-16_14-00-11__43__1.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-03-25_13-00-43__29__16.mkv': {'GT_TotalIn': 26, 'Pre__TotalIn': 26, 'GT_Current': 13, 'Pred_Current': 13}}



sortOH with yolo center
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 9, 'GT_Current': -6, 'Pred_Current': -7}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': -5, 'Pred_Current': -6}}計少一個
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-05-08_16-00-46__8__0.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 7, 'GT_Current': 0, 'Pred_Current': -1}}

{'usedVideos/cam1-2022-03-25_13-00-43__29__16.mkv': {'GT_TotalIn': 29, 'Pre__TotalIn': 26, 'GT_Current': 16, 'Pred_Current': 14}}
{'usedVideos/cam1-2022-04-16_14-00-11__43__2.mkv': {'GT_TotalIn': 43, 'Pre__TotalIn': 51, 'GT_Current': 2, 'Pred_Current': 5}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 37, 'GT_Current': 4, 'Pred_Current': 3}}

sortOH merge
'oneVideo35/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 37, 'GT_Current': 4, 'Pred_Current': -1}}
'oneVideo35/cam1-2022-03-25_13-00-43__29__16.mkv': {'GT_TotalIn': 29, 'Pre__TotalIn': 26, 'GT_Current': 16, 'Pred_Current': 13}}
'oneVideo35/cam1-2022-04-16_14-00-11__43__2.mkv': {'GT_TotalIn': 43, 'Pre__TotalIn': 58, 'GT_Current': 2, 'Pred_Current': 9}}

centroid
In:90/103 out:65/90

{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 4, 'GT_Current': 1, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 8, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 3}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 5, 'GT_Current': -5, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 15, 'GT_Current': 1, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_16-00-46__8__0.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 7, 'GT_Current': 0, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 11, 'GT_Current': -2, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 8, 'GT_Current': 1, 'Pred_Current': -3}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 14, 'GT_Current': -6, 'Pred_Current': 3}}


66/104 54/83

{'usedVideos/cam1-2022-04-16_14-00-11__43__2.mkv': {'GT_TotalIn': 43, 'Pre__TotalIn': 67, 'GT_Current': 2, 'Pred_Current': 15}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 47, 'GT_Current': 4, 'Pred_Current': 12}}
{'usedVideos/cam1-2022-03-25_13-00-43__29__16.mkv': {'GT_TotalIn': 29, 'Pre__TotalIn': 28, 'GT_Current': 16, 'Pred_Current': 21}}


deepsort


{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 2, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-05-08_16-00-46__8__0.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 8, 'GT_Current': 2, 'Pred_Current': 3}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 9, 'GT_Current': 1, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': -5, 'Pred_Current': -6}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -3}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 14, 'GT_Current': 1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}



{'usedVideos/cam1-2022-03-25_13-00-43__29__16.mkv': {'GT_TotalIn': 29, 'Pre__TotalIn': 27, 'GT_Current': 16, 'Pred_Current': 20}}
{'usedVideos/cam1-2022-04-16_14-00-11__43__2.mkv': {'GT_TotalIn': 43, 'Pre__TotalIn': 53, 'GT_Current': 2, 'Pred_Current': 8}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 34, 'GT_Current': 4, 'Pred_Current': 0}}



sortOH merge
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 4, 'GT_Current': 2, 'Pred_Current': 3}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -3}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 3}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': -4}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': -5, 'Pred_Current': -8}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-05-08_16-00-46__8__0.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 7, 'GT_Current': 0, 'Pred_Current': -2}}

{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 36, 'GT_Current': 4, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-16_14-00-11__43__2.mkv': {'GT_TotalIn': 43, 'Pre__TotalIn': 54, 'GT_Current': 2, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-03-25_13-00-43__29__16.mkv': {'GT_TotalIn': 29, 'Pre__TotalIn': 26, 'GT_Current': 16, 'Pred_Current': 13}}


sortOH with merge out

{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': -3}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -3}}
{'usedVideos/cam1-2022-05-08_16-00-46__8__0.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 7, 'GT_Current': 0, 'Pred_Current': -3}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -6}}
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': -5, 'Pred_Current': -9}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 36, 'GT_Current': 4, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': -4}}
{'usedVideos/cam1-2022-03-25_13-00-43__29__16.mkv': {'GT_TotalIn': 29, 'Pre__TotalIn': 26, 'GT_Current': 16, 'Pred_Current': 13}}
{'usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 0}}



sortOH with merge out
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 17, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -6}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': -5, 'Pred_Current': -6}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 12, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-08_16-00-46__8__0.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 7, 'GT_Current': 0, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': -1}}

{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 38, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-03-25_13-00-43__29__16.mkv': {'GT_TotalIn': 29, 'Pre__TotalIn': 26, 'GT_Current': 16, 'Pred_Current': 13}}


sortOH with merge out area

{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -6}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 17, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 12, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': -5, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__-1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-05-08_16-00-46__7__-1.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': -1, 'Pred_Current': -1}}


{'usedVideos/cam1-2022-03-25_13-00-43__23__13.mkv': {'GT_TotalIn': 23, 'Pre__TotalIn': 26, 'GT_Current': 13, 'Pred_Current': 13}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 38, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 2}}



count out area should change
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__-1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -1, 'Pred_Current': 0}}merge distance decrease current should be 1
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}one not count in, sortOH think is same person
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': 1}}106 should not spilt as flash ID
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 3}}flash id threshold change to 15, 8??
{'usedVideos/cam1-2022-04-16_16-00-17__7__-5.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': -5, 'Pred_Current': -6}}
{'usedVideos/cam1-2022-05-08_16-00-46__7__-1.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': -1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': -1}}

{'usedVideos/cam1-2022-03-25_13-00-43__26__14.mkv': {'GT_TotalIn': 26, 'Pre__TotalIn': 26, 'GT_Current': 14, 'Pred_Current': 14}}
{'usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 3}}

{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 36, 'GT_Current': 4, 'Pred_Current': 2}}



{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__-1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -1, 'Pred_Current': 0}}}merge distance decrease current should be 1
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_16-00-46__7__-1.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': -1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-16_16-00-17__6__-6.mkv': {'GT_TotalIn': 6, 'Pre__TotalIn': 6, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': -2}}


{'usedVideos/cam1-2022-03-25_13-00-43__26__14.mkv': {'GT_TotalIn': 26, 'Pre__TotalIn': 26, 'GT_Current': 14, 'Pred_Current': 14}}
{'usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 36, 'GT_Current': 4, 'Pred_Current': 1}}merge id area should change




sortOH with merge out
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-16_16-00-17__6__-6.mkv': {'GT_TotalIn': 6, 'Pre__TotalIn': 6, 'GT_Current': -6, 'Pred_Current': -5}}merge area should change or merge distance increase
{'usedVideos/cam1-2022-05-08_16-00-46__7__-1.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': -1, 'Pred_Current': -1}}


{'usedVideos/cam1-2022-03-25_13-00-43__26__14.mkv': {'GT_TotalIn': 26, 'Pre__TotalIn': 26, 'GT_Current': 14, 'Pred_Current': 14}}
{'usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 36, 'GT_Current': 2, 'Pred_Current': 2}}




sortOH merge out without flashID checking
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-16_16-00-17__6__-6.mkv': {'GT_TotalIn': 6, 'Pre__TotalIn': 6, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__4.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 36, 'GT_Current': 4, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-03-25_13-00-43__26__14.mkv': {'GT_TotalIn': 26, 'Pre__TotalIn': 26, 'GT_Current': 14, 'Pred_Current': 14}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_16-00-46__7__-1.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': -1, 'Pred_Current': -1}}




sortOH with count area changed and out in
{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__2.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 36, 'GT_Current': 2, 'Pred_Current': 3}}
{'usedVideos/cam1-2022-04-16_16-00-17__6__-6.mkv': {'GT_TotalIn': 6, 'Pre__TotalIn': 6, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-03-25_13-00-43__26__14.mkv': {'GT_TotalIn': 26, 'Pre__TotalIn': 26, 'GT_Current': 14, 'Pred_Current': 14}}
{'usedVideos/cam1-2022-04-03_15-00-21__1__1.mkv': {'GT_TotalIn': 1, 'Pre__TotalIn': 26, 'GT_Current': 1, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_16-00-46__7__-1.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': -1, 'Pred_Current': -1}}




{'usedVideos/cam1-2022-04-13_15-00-26__3__2.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-04-02_15-00-07__3__1.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-03_16-00-24__8__4.mkv': {'GT_TotalIn': 8, 'Pre__TotalIn': 8, 'GT_Current': 4, 'Pred_Current': 4}}
{'usedVideos/cam1-2022-04-03_14-00-19__16__1.mkv': {'GT_TotalIn': 16, 'Pre__TotalIn': 16, 'GT_Current': 1, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-10_15-00-46__10__-2.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -2, 'Pred_Current': -2}}
{'usedVideos/cam1-2022-04-10_14-00-44__10__-6.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-03-25_13-00-43__26__14.mkv': {'GT_TotalIn': 26, 'Pre__TotalIn': 26, 'GT_Current': 14, 'Pred_Current': 14}}
{'usedVideos/cam1-2022-04-16_15-00-14__35__2.mkv': {'GT_TotalIn': 35, 'Pre__TotalIn': 36, 'GT_Current': 2, 'Pred_Current': 1}}
{'usedVideos/cam1-2022-04-16_14-00-11__47__2.mkv': {'GT_TotalIn': 47, 'Pre__TotalIn': 50, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-07_17-00-36__3__0.mkv': {'GT_TotalIn': 3, 'Pre__TotalIn': 3, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-04-16_16-00-17__6__-6.mkv': {'GT_TotalIn': 6, 'Pre__TotalIn': 6, 'GT_Current': -6, 'Pred_Current': -5}}
{'usedVideos/cam1-2022-05-07_14-00-27__11__0.mkv': {'GT_TotalIn': 11, 'Pre__TotalIn': 11, 'GT_Current': 0, 'Pred_Current': -1}}
{'usedVideos/cam1-2022-05-08_14-00-40__7__0.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 6, 'GT_Current': 0, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-07_16-00-33__7__2.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': 2, 'Pred_Current': 2}}
{'usedVideos/cam1-2022-05-08_15-00-43__10__1.mkv': {'GT_TotalIn': 10, 'Pre__TotalIn': 10, 'GT_Current': 1, 'Pred_Current': 0}}
{'usedVideos/cam1-2022-05-08_16-00-46__7__-1.mkv': {'GT_TotalIn': 7, 'Pre__TotalIn': 7, 'GT_Current': -1, 'Pred_Current': -1}}