#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.network = None
        self.core = None
        self.input_blob = None
        self.out_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model_xml, device='CPU', cpu_extension=None):
        ### TODO: Load the model ###
        self.network = IENetwork(model=model_xml, weights=model_xml[:-3] + 'bin')
        
        # Initialise Core
        self.core = IECore()
        if cpu_extension and "CPU" in device:
            self.core.add_extension(cpu_extension, device)
            
        # attach network to core
        self.exec_network = self.core.load_network(self.network, device)
        
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def init_async_infer(self,image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        wait_=self.exec_network.requests[0].wait(-1)
        return wait_

    def get_output(self):
        return self.exec_network.requests[0].outputs[self.output_blob]
