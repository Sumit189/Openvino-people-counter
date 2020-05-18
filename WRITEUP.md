Project Write-Up

python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels

Explaining Custom Layers
The layers which are not available in OpenVino framework known to be custom layers. To convert a custom layer the process is to load or register the custom layer as an extension to the model optimizer. For TensorFlow and other frameworks the process can differ. The need of handling the custom layers is important otherwise model optimizer wont recognize it and hence IR can't be obtained. Many layers are not supported by OpenVino for that layers extension should be added to the model optimizer.

Comparing Model Performance
My method(s) to compare models before and after conversion to Intermediate Representations were by running the model on same video and checking the following things:

How accurately the people are getting detected in the frame pre and post conversion i.e if a person is getting left out of the detection even after lowering the threshold then it is not good
Secondly checking the speed of the inference on the basis how fast the frame is getting returned.
The difference between model accuracy pre- and post-conversion was dropped as I was reduced certainly while using the MobileNet SSD models but when I tried the Faster RCNN Inception v2 and SSD Resnet the accuracy was certain but the inference time was affected a lot.

Assess Model Use Cases
Some of the potential use cases of the people counter app are:

It can used to solve the problem of lack of supplies for the number of the customers in the shop.
It can be used for CCTV footage becuase it can avoid the recording of non-suspecious activities which is beneficial for database.
Crowd management.
These use cases are useful because if these activities are automated it would create a great impact/

Assess Effects on End User Needs
Model accuracy: It needs to be good enough to correctly complete most of the task without giving much error becuase it needs to be used in the area where it would act all by itself so true/correct results are required.
Length, width if image or video produced by the camera: The models are trained on some specefic shape of image like 256x256, or 128x128. If a camera is producing an image with size more than the shape than it needs to be prprocessed before passing to the model for inference, it can increase time and will require little computation power for reshaping, beneficial way would be to get the image of required size directly from camera.

Model Research
In investigating potential people counter models, I tried each of the following three models:

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
