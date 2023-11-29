#!/usr/bin/python

import json
import os
import sys


class SSIM_test_analyze():
    def __init__(self, folder, start_thresh=50, final_thresh=200, interval_thresh=25,
                 saving_path="", display_message=False):     
        
        """
          Initialize input parameters.
        """
        self.SSIM_TP = None
        self.SSIM_FP = None
        self.SSIM_FN = None
        self.all_luminance_cnt = None
        self.yolo_luminance_cnt = None
        self.folder = folder
        self.start_thresh = start_thresh
        self.final_thresh = final_thresh
        self.interval_thresh = interval_thresh
        self.saving_path = saving_path
        self.display_message = display_message
        
    
    def analyze(self):        
        """
          Analyze the SSIM detection results (TP, FP, FN)
          to find the optimal threshold.
        """
        
        if not os.path.exists(self.saving_path):
            self.create_dir(self.saving_path)
        
        # Check the file path and load the data
        if isinstance(self.folder, str):
            self.SSIM_TP, self.SSIM_FP,  self.SSIM_FN, \
            self.all_luminance_cnt, self.yolo_luminance_cnt = \
            self.check_folder_exists(self.folder)        
        elif isinstance(self.folder, list):
            self.SSIM_TP, self.SSIM_FP, self.SSIM_FN, \
            self.all_luminance_cnt, self.yolo_luminance_cnt = \
            self.SSIM_test_results_integrate(self.folder)
        else:
            raise AssertionError("[Error] The input folder must be string or list.")
        

        """
          Initalize parameters.
        """
        # Record SSIM optimal threshold
        self.optimal_threshold_precision_dict = {}
        self.optimal_threshold_recall_dict = {}
        
        all_TP = {} # Each SSIM threshold TP with all luminance
        all_FP = {} # Each SSIM threshold FP with all luminance
        all_FN = {} # Each SSIM threshold FN with all luminance
        
        for t in range(self.start_thresh, self.final_thresh, 
                           self.interval_thresh):
            all_TP[t] = 0 # Get each threshold TP in all luminance
            all_FP[t] = 0 # Get each threshold FP in all luminance
            all_FN[t] = 0 # Get each threshold FN in all luminance
        
        
        """
          Interate all luminance.
        """
        for lum in range(255):
            lum = str(lum) 
            
            best_precision = {} # Record different SSIM threshold precision in this luminance
            best_recall = {} # Record different SSIM threshold recall in this luminance
            
            
            """
              Get different SSIM threshold results in this luminance.
            """
            for t in range(self.start_thresh, self.final_thresh, 
                           self.interval_thresh):
                
                all_TP[t] += self.SSIM_TP[str(t)][lum]
                all_FP[t] += self.SSIM_FP[str(t)][lum]
                all_FN[t] += self.SSIM_FN[str(t)][lum]

            """
              Compute optimal SSIM threshold of Precision and Recall in this luminance. 
            """
            if self.SSIM_TP[str(t)][lum] + self.SSIM_FP[str(t)][lum] == 0:
                precision = -1
            else:
                precision = self.SSIM_TP[str(t)][lum] / (self.SSIM_TP[str(t)][lum] + self.SSIM_FP[str(t)][lum])

            if (self.SSIM_TP[str(t)][lum] + self.SSIM_FN[str(t)][lum]) == 0:
                recall = -1
            else:
                recall = self.SSIM_TP[str(t)][lum] / (self.SSIM_TP[str(t)][lum] + self.SSIM_FN[str(t)][lum])           

            best_precision[t] = precision
            best_recall[t] = recall
            
            # Optimal SSIM threshold (each luminance)
            self.print_msg(f"Lum:{lum}, Number of images:{self.all_luminance_cnt[str(lum)]}, Number of images yolo detected:{self.yolo_luminance_cnt[str(lum)]}")
            optimal_threshold_precision, optimal_threshold_recall = self.get_optimal_threshold(best_precision, best_recall)
            self.optimal_threshold_precision_dict[lum] = optimal_threshold_precision
            self.optimal_threshold_recall_dict[lum] = optimal_threshold_recall
            self.print_msg("."*50)
        
        """
          Compute each threshold Precision and Recall.
        """
        best_precision = {}
        best_recall = {}
        
        for t in range(self.start_thresh, self.final_thresh, 
                           self.interval_thresh):            

            if all_TP[t] + all_FP[t] == 0:
                precision = -1
            else:
                precision = all_TP[t]/(all_TP[t] + all_FP[t])

            if (all_TP[t] + all_FN[t]) == 0:
                recall = -1
            else:
                recall = all_TP[t] / (all_TP[t] + all_FN[t])           

            best_precision[t] = precision
            best_recall[t] = recall
            
            # Each threshold precision and recall (depend on all images)
            self.print_msg(f"Threshold:{t}, Average Precision={precision}, Average Recall={recall}")
            self.print_msg(f"TP:{all_TP[t]}, FP:{all_FP[t]} FN:{all_FN[t]}")
            self.print_msg("-"*50)

        # Overall SSIM optimal threshold (depend on all images)
        self.print_msg("Optimal threshold in all luminance:")
        self.get_optimal_threshold(best_precision, best_recall)
        self.print_msg("="*50)
        
        
        """
          Write each luminance SSIM threshold to text file.
        """       
        
            
        with open(os.path.join(self.saving_path, "SSIM_optimal_threshold_precision.txt"), "w") as f:
            f.write(json.dumps(self.optimal_threshold_precision_dict))
            
        with open(os.path.join(self.saving_path, "SSIM_optimal_threshold_recall.txt"), "w") as f:
            f.write(json.dumps(self.optimal_threshold_recall_dict))
            
        
    
    
    def get_optimal_threshold(self, best_precision: list, best_recall: list):
        """
          Sort and get SSIM optimal threshold of precision and recall.
        """
        average_precision = 0
        average_recall = 0
        
        # Get optimal threshold of precision
        value = sorted(set(best_precision.values()))[-1]
        if value == -1:
            self.print_msg("SSIM Optimal threshold (precision): Null")
        else:        
            thresh = {i for i in best_precision if best_precision[i]==value}
            thresh = list(thresh)            
            
            for t in thresh:
                self.print_msg(f"SSIM Optimal threshold (precision):{t} Precision: {best_precision[t]} (Recall:{best_recall[t]})")
                average_precision += t
                
            average_precision /= len(thresh)
        
        
        # Get optimal threshold of recall
        value = sorted(set(best_recall.values()))[-1]
        if value == -1:
            self.print_msg("SSIM Optimal threshold (recall): Null")
        else:
            thresh = {i for i in best_recall if best_recall[i]==value}
            thresh = list(thresh)            
            
            for t in thresh:
                self.print_msg(f"SSIM Optimal threshold (recall):{t} Recall: {best_recall[t]} (Precision:{best_precision[t]})")
                average_recall += t
                
            average_recall /= len(thresh)            
        
        return int(average_precision), int(average_recall)
        
        
    def SSIM_test_results_integrate(self, folder: list):
        """
          Integrate the SSIM_test results.          
        """
       
        # Initialize parameters.      
        all_luminance_cnt = {}
        yolo_luminance_cnt = {}
        SSIM_TP = {}
        SSIM_FP = {}
        SSIM_FN = {}
        
        for v in range(255):
            all_luminance_cnt[str(v)] = 0
            yolo_luminance_cnt[str(v)] = 0

        for i in range(self.start_thresh, 
                       self.final_thresh, 
                       self.interval_thresh):
            tmp = {}
            for j in range(255):
                tmp[str(j)] = 0

            SSIM_TP[str(i)] = tmp.copy()
            SSIM_FP[str(i)] = tmp.copy()
            SSIM_FN[str(i)] = tmp.copy()
        
        
        """
          Integrate each folder results.
        """
        for folder_path in folder:
            SSIM_TP_tmp, SSIM_FP_tmp, SSIM_FN_tmp, \
            all_luminance_cnt_tmp, yolo_luminance_cnt_tmp = \
            self.check_folder_exists(folder_path)
            
            for v in range(255):
                all_luminance_cnt[str(v)] += all_luminance_cnt_tmp[str(v)]
                yolo_luminance_cnt[str(v)] += yolo_luminance_cnt_tmp[str(v)]
                
            for i in range(self.start_thresh, 
                       self.final_thresh, 
                       self.interval_thresh):
                
                for j in range(255):
                    SSIM_TP[str(i)][str(j)] += SSIM_TP_tmp[str(i)][str(j)]
                    SSIM_FP[str(i)][str(j)] += SSIM_FP_tmp[str(i)][str(j)]
                    SSIM_FN[str(i)][str(j)] += SSIM_FN_tmp[str(i)][str(j)]
            
        """
          Saving Integrate SSIM_test results.
        """
        with open(os.path.join(self.saving_path, "TP_integrate.txt"), "w") as f:
            f.write(json.dumps(SSIM_TP))
            
        with open(os.path.join(self.saving_path, "FP_integrate.txt"), "w") as f:
            f.write(json.dumps(SSIM_FP))
            
        with open(os.path.join(self.saving_path, "FN_integrate.txt"), "w") as f:
            f.write(json.dumps(SSIM_FN))
            
        with open(os.path.join(self.saving_path, "all_luminance_cnt_integrate.txt"), "w") as f:
            f.write(json.dumps(all_luminance_cnt))
            
        with open(os.path.join(self.saving_path, "yolo_luminance_cnt_integrate.txt"), "w") as f:
            f.write(json.dumps(yolo_luminance_cnt))
            
       
        return SSIM_TP, SSIM_FP, SSIM_FN, all_luminance_cnt, yolo_luminance_cnt
    
    
    def check_folder_exists(self, folder):        
        TP_path = os.path.join(folder, "TP.txt")
        FP_path = os.path.join(folder, "FP.txt")
        FN_path = os.path.join(folder, "FN.txt")
        all_luminance_cnt_path = os.path.join(folder, "all_luminance_cnt.txt")
        yolo_luminance_cnt_path = os.path.join(folder, "yolo_luminance_cnt.txt")
        
        if not os.path.exists(TP_path):
            raise AssertionError(f"[Error] TP_path not exist:{TP_path}")
        
        if not os.path.exists(FP_path):
            raise AssertionError(f"[Error] FP_path not exist:{FP_path}")
            
        if not os.path.exists(FN_path):
            raise AssertionError(f"[Error] FN_path not exist:{FN_path}")
            
        if not os.path.exists(all_luminance_cnt_path):
            raise AssertionError(f"[Error] all_luminance_cnt_path not exist:{all_luminance_cnt_path}")      

        if not os.path.exists(yolo_luminance_cnt_path):
            raise AssertionError(f"[Error] yolo_luminance_cnt_path not exist:{yolo_luminance_cnt_path}")
            
        with open(TP_path) as file:
            data = file.read()        
            SSIM_TP = json.loads(data) 

        with open(FP_path) as file:
            data = file.read()        
            SSIM_FP = json.loads(data)

        with open(FN_path) as file:
            data = file.read()        
            SSIM_FN = json.loads(data) 

        with open(all_luminance_cnt_path) as file:
            data = file.read()        
            all_luminance_cnt = json.loads(data)

        with open(yolo_luminance_cnt_path) as file:
            data = file.read()        
            yolo_luminance_cnt = json.loads(data)
        
        
        return SSIM_TP, SSIM_FP, SSIM_FN, all_luminance_cnt, yolo_luminance_cnt
        
    
    def print_msg(self, msg: str):        
        if self.display_message:            
            print(msg)
            
    
    def create_dir(self, output_dir: str):   
        try:
            path = os.path.join(output_dir)
            os.makedirs(path, exist_ok=True)
            return path

        except OSError as e:
            print("Create dirictory error:",e)
            
            
if __name__ == "__main__":   
    
    """
      Read the SSIM_test folder path to generate the SSIM optimal threshold file.
      Args: 
        - folder
            input type:
              - string: SSIM_test path results.
              - list: All folder path want to integrate SSIM_test results.
        - saving_path: SSIM optimal threshold results path want to save.
          (Auto generate folder if not exist)
    """
    
    folder = []
    dir_path = "/home/jim93073/YoloTalk/test_SSIM/EECS/"
    for f in os.listdir(dir_path):
        if "ipynb" not in f:
            folder.append(os.path.join(dir_path, f, "SSIM_test"))
    
    
    SSIM_results = SSIM_test_analyze(folder = folder,
                                     saving_path = dir_path,
                   )    
    SSIM_results.analyze()
    
