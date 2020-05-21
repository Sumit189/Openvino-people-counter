# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

Deep learning became trending in 21 century because of the high computation power and a large amount of data to feed the neural networks. Depending on the requirements of the application many layers are combined to form a powerful model. This helps to create a new model for Deep learning or multiple operations using custom layers.

The process behind converting custom layers involves the following. 
* First of all, Generate the Extension Template Files Using the Model Extension Generator that is provided in OpenVino
* Use the provided Model Optimizer to Generate IR Files Containing the Custom Layer
* Edit the CPU Extension Template Files
* Execute the Model with the Custom Layer

There are two most popular models for pose detection that are YOLO (You Only Look Once) and SSD (Single Shot multi-box Detector) model. The thing that makes both of them different from each other is the accuracy of the model and the computational power. YOLO is suitable for high speed but it has a downside of accuracy whereas the SSD model has higher accuracy but it cost large computational time. In the process of creating a custom layer, I choose the SSD MobileNet V2 COCO model.

* First download the SSD MobileNet V2 COCO model from Tensorflow using Wget command

```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```

*Generating IR

* Use tar -xvf command and to unpack it.

```
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```

* To convert the TensorFlow model, we have to feed the downloaded SSD MobileNet V2 COCO model's .pb file with the help of model optimizer.
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

 If the conversion is successful, there will be a .xml file and the .bin file. The Execution Time is around 80 to 90 seconds.

The Generated IR model files are : 
* XML file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.xml
* BIN file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.bin


### To run this project, use the following commands:

####Video file

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

#### Camera stream 
```
python main.py -i CAM -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were the following.: 

The difference between the accuracy of the model got from pre-conversion and post-conversion is that The SSD model (MobileNet V2 COCO model) Intermediate representation was able to detect less number of people in the frame but with high accuracy but it fails to continuously keep track of subject that is not in motion.

The size of the model pre- and post-conversion was almost the same. The SSD MobileNet V2 COCO model .pb file is about 66.4 MB and the IR bin file is 64.1 MB. 

The inference time of the model pre- and post-conversion was 70ms. I tested both the pre-trained model and the converted model, where it turns out that, the pre-trained model from the open zoo had a lesser inference time that the converted model. Also, the detection was so accurate with the pre-trained model. 

## Assess Model Use Cases

It can be used to solve the problem of the lack of supplies for the number of customers in the shop.

It can be used for CCTV footage because it can avoid the recording of non-suspicious activities which is beneficial for the database.

Crowd management on the street.

Each of these use cases would be useful because, it allows us to improve the marketing strategy of the retail and as well as safety of the pedestrian.

## Assess Effects on End-User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows. 

* Lighting and Focal Length of the camera depends on the system installed. Bad lighting can seriously reduce the accuracy of the model because the model will not able to find the patterns in the new input. 

* The angle of the camera plays an important role. It can seriously affect lighting conditions and model accuracy. 

* The camera image size should be compatible with the model for proper detection. The model accuracy is calculated using the confusion matrix which gives the details about the occurrence of false positives and negatives which degrades the accuracy of the model. 

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people countermodels, I tried each of the following three models:

Model 1: SSD_RESNET50_V1_FPN_COCO

http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
I converted the model to an Intermediate Representation with the following arguments
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel
The loading time for inference make this model non usable for the application.

Model 2: FASTER RCNN INCEPTION v2

http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
I converted the model to an Intermediate Representation with the following arguments
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --reverse_input_channel
It alsi had same problem of taking long time in loading/inferencing on a frame, which also make it less usable if we want results faster.

Model 3: SSD MOBILENET V1 FPN COCO

http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
I converted the model to an Intermediate Representation with the following arguments
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --tensorflow_object_detection_api_pipeline_config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel
The accuracy got altered in poor way. after reducing the threshold value. If the accuracy is bot good enough the model cannot be used for production.

