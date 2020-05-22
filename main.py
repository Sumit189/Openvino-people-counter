"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt
from collections import deque
from argparse import ArgumentParser
from inference import Network
import math
# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client=mqtt.Client() 
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_box(coords,frame,initial_w,initial_h,x,k):
    current_count=0
    ed=x
    for obj in coords[0][0]:
        if obj[2]>prob_threshold:
            xmin=int(obj[3]*initial_w)
            ymin=int(obj[4]*initial_h)
            xmax=int(obj[5]*initial_w)
            ymax=int(obj[6]*initial_h)
            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,0,255),1)
            current_count+=1

            c_x=frame.shape[1]/2
            c_y=frame.shape[0]/2
            mid_x=(xmax+xmin)/2
            mid_y=(ymax+ymin)/2

            ed=math.sqrt(math.pow(mid_x-c_x,2)+math.pow(mid_y-c_y,2)*1.0)
            k=0
    
    if current_count<1:
        k+=1
    if ed>0 and k<10:
        current_count=1
        k+=1
        if k>100:
            k=0
    
    return frame, current_count,ed,k

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    args=build_argparser().parse_args()
    single_image_mode=False
    
    # Initialise the class
    infer_network = Network()
    model=args.model
    video_file=args.input
    extnsn=args.cpu_extension
    device=args.device
   

    start_time=0
    cur_request_id=0
    last_count=0
    total_count=0

    n,c,h,w=infer_network.load_model(model,device,1,1,cur_request_id,extnsn)[1]

    ### TODO: Handle the input stream ###
        # Checks for live feed
    if video_file == 'CAM':
        input_stream= 0

    # Checks for input image
    elif video_file.endswith('.jpg') or video_file.endswith('.bmp') :
        single_image_mode = True
        input_stream=video_file

    else:
        input_stream=video_file
        assert os.path.isfile(video_file), "File doesn't exist"
    
    try:   
        # Capture video
        capture=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate the file: "+video_file)
    except Exception as e:
        print("Something went wrong with the file: "+e)
    

    global initial_w,initial_h,prob_threshold
    total_count=0
    duration=0
    initial_w=capture.get(3)
    initial_h=capture.get(4)
     # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    temp=0
    tk=0
    #Loop until stream is over
    while capture.isOpened():
        flag,frame=capture.read()
        if not flag:
            break
        
        key_pressed=cv2.waitKey(60)
        
        #Pre-processing the input/frame
        image=cv2.resize(frame,(w,h))
        image=image.transpose((2,0,1))
        image.reshape((n,c,h,w))
        
        #Async inference
        inf_start=time.time()
        infer_network.exec_net(cur_request_id,image)
        color=(255,0,0)

        #Waiting for result
        if infer_network.wait(cur_request_id)==0:
            time_elapsed=time.time()-inf_start
            
            #Result from the inference
            result=infer_network.get_output(cur_request_id)
            
            #Bounting box
            frame, current_count,d,tk=draw_box(result,frame,initial_w,initial_h,temp,tk)
            
            #inference time
            inf_timemsg="Inference Time: {:,3f}ms".format(time_elapsed*1000)
            cv2.putText(frame,inf_timemsg,(15,15),cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

            #Calculating and sending info
            if current_count>last_count:
                start_time=time.time()
                total_count=total_count+current_count-last_count
                client.publish("person",json.dumps({"total":total_count}))
            
            if current_count<last_count:
                duration=int(time.time()-start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))

            text_2="Distance: %d" %d+" Lost frame: %d" %tk   
            cv2.putText(frame,text_2,(15,30),cv2.FONT_HERSHEY_COMPLEX,0.5, color,1)

            text_2="Current count: %d" %current_count   
            cv2.putText(frame,text_2,(15,45),cv2.FONT_HERSHEY_COMPLEX,0.5, color,1)

            if current_count>3:
                text_2="Maximum count reached!!!"
                (text_width,text_height)=cv2.getTextSize(text_2,cv2.FONT_HERSHEY_COMPLEX,0.5, thickness=1)[0]
                text_offset_x=10
                text_offset_y=frame.shape[0]-10
                box_coords = ((text_offset_x, text_offset_y + 2), (text_offset_x + text_width, text_offset_y - text_height - 2))
                cv2.rectangle(frame,box_coords[0],box_coords[1],(0,0,0),cv2.FILLED)
                cv2.putText(frame, text_2, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            
            client.publish("person",json.dumps({"count":current_count}))

            last_count=current_count
            temp=d
            if key_pressed==27:
                break

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        #Saving Image
        if single_image_mode:
            cv2.write('output_image.jpg',frame)

    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
