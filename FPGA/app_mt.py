'''
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
from vaitrace_py import vai_tracepoint
xrt_profile = True
divider='---------------------------'

@vai_tracepoint
def preprocess_fn(image_path):
    '''
    Image pre-processing.
    Opens image as grayscale then normalizes to range 0:1
    input arg: path of image file
    return: numpy array
    '''
    #print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(224,224))
    img = img/255
    b,g,r = cv2.split(img)
    
    mean = [0.485,0.456,0.406]
    std = [0.229,0.224,0.225]
    
    b = (b - mean[2])/std[2]
    g = (g - mean[1])/std[1]
    r = (r - mean[0])/std[0]
    #print(b)
    #sys.exit()
    #b = b / std[2] 
    #g = g / std[1]
    #r = r / std[0]
    
    #for i in range(224):
    #    for j in range(224):
    #        b[i][j] = b[i][j] - 1.8044
            
    #for i in range(224):
    #    for j in range(224):
    #        g[i][j] = g[i][j] - 2.0357
            
    #for i in range(224):
    #    for j in range(224):
    #        r[i][j] = r[i][j] - 2.1179
    #for i in range(len(b)):
    #    for j in range(len(b[0])):
    #        b[i][j] = (b[i][j] - mean[2])/std[2]
            
    #for i in range(len(g)):
    #    for j in range(len(g[0])):
    #        g[i][j] = (g[i][j] - mean[1])/std[1]
            
    #for i in range(len(r)):
    #    for j in range(len(r[0])):
    #        r[i][j] = (r[i][j] - mean[0])/std[0]
            
            
    #sys.exit()
    
    
    image = cv2.merge([r,g,b])
    #print(image.shape)
    #sys.exit()
    #image = image.reshape(224,224,3)
    #print("image.shape",image.shape)
    
    #image = image[:,:,::-1].transpose((2,0,1))
    #print("image_dim",image.shape)
    
    #print(image)
    #image = image/255
    
    return image


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

@vai_tracepoint
def runDPU(id,start,dpu,img):

    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    print(inputTensors)
    print(outputTensors)
    #sys.exit()
    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = 0
    print("runDPU")
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        '''init input image to input buffer '''
        #for j in range(runSize):
        imageRun = inputData[0]
            #imageRun[j, ...] = img[(count + j) % n_of_images]#.reshape(input_ndim[1:])
        imageRun[0, ...] = img[count]
        #print(inputData,"input")
        #sys.exit()
        '''run with batch '''
        #inputData[0] = img[count]
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)

        '''store output vectors '''
        for j in range(runSize):
            #print(outputData[0][j])
            #sys.exit()
            #print("length",len(outputData[0][j]))
            #out_q[write_index] = np.argmax(outputData[0][0])
            out_q[write_index] = np.argmax(outputData[0][0])
        write_index += 1
        count = count + 1

@vai_tracepoint
def app(image_dir,threads,model):

    #listimage=os.listdir(image_dir)
    
    list_path = np.load('./20210812imagenet_val.npy')
    #list_path = list_path[0:1000]
    #print(list_path)
    listimage = []
    truth_label = []
    for i in range(1000):
        #print(i)
        listimage.append('./ImageNet100/'+list_path[i][0]+'/'+list_path[i][1])
        truth_label.append(list_path[i][2])
    #print(listimage)
    runTotal = len(listimage)
    #sys.exit()
    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    ''' preprocess images '''
    print('Pre-processing',runTotal,'images...')
    img = []
    a = 0
    for i in range(runTotal):
        #path = os.path.join(image_dir,listimage[i])
        path = listimage[i]
        #print(path)
        img.append(preprocess_fn(path))
        #a = a + 1
        #if a % 100 == 10:
        #    print(a)

    '''run threads '''
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))
    #sys.exit()
    ''' post-processing '''
    #classes = ['zero','one','two','three','four','five','six','seven','eight','nine'] 
    classes = [str(i) for i in range(18)]
    #print("classes",classes)
    correct = 0
    wrong = 0
    #print("out_q",out_q)
    #sys.exit()
    #print(out_q)
    for i in range(len(out_q)):
        #prediction = classes[out_q[i]]
        #prediction = out_q[i]
        #print(prediction)
        
        #ground_truth, _ = listimage[i].split('_',1)
        #print("label",truth_label[i],out_q[i])
        #print(type(truth_label[i]),type(np.str(out_q[i])))
        if (truth_label[i]==np.str(out_q[i])):
            
            correct += 1
        else:
            wrong += 1
    accuracy = correct/len(out_q)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' %(correct,wrong,accuracy))
    print(divider)

    return




# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='./ImageNet100/U+9ED2_3', help='Path to folder of images. Default is images')  
  ap.add_argument('-t', '--threads',   type=int, default=2,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='resprun.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')

  args = ap.parse_args()  
  print(divider)
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)
  print(divider)

  app(args.image_dir,args.threads,args.model)

if __name__ == '__main__':
  main()

