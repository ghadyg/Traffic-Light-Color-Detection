from Detection import *

detector =Detector()
classFile ="coco.names"

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
detector.readClasses(classFile)

detector.downloadModel(modelURL)
detector.loadModel()
i1=r"images\image1.jpe"
i2=r"images\image2.jpe"
i3=r"images\full.jpg"
i4=r"images\traffic-light-red.jpe"
i5=r"images\yellow.jpe"
i6=r"images\image4.jpg"
image=[i1,i2,i3,i4,i5,i6]
for im in image:
    
    detector.detectColor(im)



