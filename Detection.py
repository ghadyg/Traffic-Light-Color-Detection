import cv2, time, os
import tensorflow as tf 
import numpy as np


from tensorflow.python.keras.utils.data_utils import get_file



class Detector:
    def __init__(self):
        pass
    def readClasses(self, classesFilePath):
        with open (classesFilePath,'r') as f:
            self.classesList = f.read().splitlines()
            

        
    def downloadModel(self,modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName =fileName[:fileName.index('.')]
        
        self.cacheDir="./pretrained_models"
        os.makedirs(self.cacheDir,exist_ok=True)
        
        get_file(fname=fileName,
        origin=modelURL,cache_dir=self.cacheDir,cache_subdir="checkpoints",extract=True)
       
    def loadModel(self):
        print("loading"+self.modelName)
        
        self.model =tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints",self.modelName,"saved_model"))
        
        print("model:"+self.modelName+" loaded succefully...")
        
        
        
    def detectColor(self,imagePath):
        image=cv2.imread(imagePath)
        
        bbox =self.createboundBox(image)
        
        for i in range(1,len(bbox)):
            
            xmin,xmax,ymin,ymax=bbox[i] 
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,255),3)
            newImage=image[ymin:ymax,xmin:xmax]
            
            #cv2.imshow("frame1",newImage)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            hsv=cv2.cvtColor(newImage,cv2.COLOR_BGR2HSV)
            ym=int(hsv.shape[0]/3)
            ymi=int(hsv.shape[0]*2/3)
            
            hsvR=hsv[:ym,:xmax]
            hsvY=hsv[ym:ymi,:xmax]
            hsvG=hsv[ymi:,:xmax]
            
            green_lower=np.array([60,134,110])
            green_upper=np.array([95,255,255])
            
            red_lower=np.array([169,130,150])
            red_upper=np.array([179,255,255])
            
            yellow_lower=np.array([10,200,130])
            yellow_upper=np.array([23,254,255])
            
            
            if(hsvG.shape[0] !=0 ):
                green_mask=cv2.inRange(hsvG,green_lower,green_upper)
                green_mask=cv2.medianBlur(green_mask, 5)
                
                green_contours,ret =cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                if green_contours!=():
                    cv2.putText(image,"GREEN",(xmin,ymin+10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            
            if(hsvY.shape[0] !=0 ):     
                yellow_mask=cv2.inRange(hsvY,yellow_lower,yellow_upper)
                yellow_mask=cv2.medianBlur(yellow_mask, 5)
                yellow_contours,ret2 =cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            
                if yellow_contours!=():
                    cv2.putText(image,"yellow",(xmin,ymin+10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),2)
            
            
            
            if(hsvR.shape[0] !=0 ):
                red_mask=cv2.inRange(hsvR,red_lower,red_upper)
                red_mask=cv2.medianBlur(red_mask, 5)
                
                red_contours,ret1 =cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                if red_contours!=():
                    cv2.putText(image,"RED",(xmin,ymin+10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            

        cv2.imshow("frame", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        
    def createboundBox(self,image):
        inputTensor=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        inputTensor=tf.convert_to_tensor(inputTensor,dtype=tf.uint8)#convert np array to tensor array
        inputTensor= inputTensor[tf.newaxis,...]
        
        detections=self.model(inputTensor) #detections is a dictionnary contains bound box,class inx and class scores
        
        bboxs=detections['detection_boxes'][0].numpy()
        classIndexes=detections['detection_classes'][0].numpy().astype(np.int32)
        classScores=detections['detection_scores'][0].numpy()
        
        imH,imW,imC =image.shape
        bboxIdx=tf.image.non_max_suppression(bboxs, classScores, 50,0.48,0.48)
        newImage= [()]
        if len(bboxIdx)!=0:
            for i in range(0,len(bboxIdx)):
                bbox=tuple(bboxs[i].tolist())
                
                classIndex=classIndexes[i]
                
                classLabelText=self.classesList[classIndex]
                
                if(classLabelText=="traffic light"):
                  
                    ymin,xmin,ymax,xmax =bbox
                    print (ymin,xmin,ymax,xmax)
                    xmin,xmax,ymin,ymax=(xmin*imW,xmax*imW,ymin*imH,ymax*imH)
                    ymin,xmin,ymax,xmax=int(ymin),int(xmin),int(ymax),int(xmax)
                    newImage=newImage.__add__([(xmin,xmax,ymin,ymax)])
                
        return newImage
                
            
                        
                
                
        
        