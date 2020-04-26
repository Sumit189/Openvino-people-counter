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

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
FRAME_KEEP=4

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

def draw_box(frame,output,prob_t,w,h):
    c_count=0
    for b in output[0][0]:
        confr=b[2]
        if confr>=prob_t:
            xmin = int(b[3] * w)
            ymin = int(b[4] * h)
            xmax = int(b[5] * w)
            ymax = int(b[6] * h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            c_count += 1
    return frame,c_count

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
 
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    ###Load the model through `infer_network` ###
    num_requests=2
    infer_network.load_model(args.model, args.device,args.cpu_extension)
    input_shape=infer_network.get_input_shape()
    n,c,h,w=input_shape
       
    ### TODO: Handle the input stream ###
        # Checks for live feed
    if args.input == 'CAM':
        args.input= 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True

    # Capture video
    capture=cv2.VideoCapture(args.input)
    capture.open(args.input)
    
    if single_image_mode:
        out=None
    else:
        out=cv2.VideoWriter('output.mp4',0x00000021,30,(100,100))
        
    if not capture.isOpened():
        log.error("Input not supported")
    
    width_=capture.get(3)
    height_=capture.get(4)
    
    
    #Init Variables
    fcounter=0
    etime=0
    c_count=0
    pcount=0
    tcount=0
    count_list=deque(maxlen=FRAME_KEEP)
    
    #Loop until stream is over
    while capture.isOpened():
        flag,frame=capture.read()
        if not flag:
            break
        
        key_pressed=cv2.waitKey(60)
        
        #Pre-processing the input/frame
        
        proc_frame=cv2.resize(frame,(w,h))
        proc_frame=proc_frame.transpose((2,0,1))
        proc_frame=proc_frame.reshape(1, *proc_frame.shape)
        
        #Async inference
        infer_network.init_async_infer(proc_frame)
        start=time.time()
        fcounter=fcounter+1
        
        #Waiting for result
        if infer_network.wait()==0:
            end=time.time()
            time_difference=end-start
            
            #Result from the inference
            result=infer_network.get_output()
            
            #Extract the desired stats from the result
            frame,c_counter=draw_box(frame,result,args.prob_threshold,width_,height_)
                #Calculate and send relevant information on current_count, total_count and duration to the MQTT server #
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person": keys of "count" and "total" ###
            message="Time: {:.3f}ms".format(time_difference*1000)
            cv2.putText(frame,message,(15,15),cv2.FONT_HERSHEY_COMPLEX,0.5,(10,200,10),1)
            count_list.append(c_count)
            average_count=sum(count_list)/4
            keep_count=int(np.ceil(average_count))
                
            if fcounter%FRAME_KEEP==0:
                if keep_count>pcount:
                    etime=time.time()
                    tcount+=(keep_count-pcount)
                    client.publish("person",json.dumps({"total":tcount}))
                if keep_count<pcount:
                    duration=int(time.time()-etime)
                    client.publish("person/duration",json.dumps({"duration":duration}))
                client.publish("person",json.dumps({"count":keep_count}))
                pcount=keep_count
            if key_pressed==27:
                break
        if not single_image_mode:
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
                
        if single_image_mode:
            cv2.write('output_img.jpg',frame)
        capture.release()
        cv2.destroyAllWindows()
        client.disconnect()


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
