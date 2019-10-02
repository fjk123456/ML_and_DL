# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import os

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

#画出混淆矩阵
def draw_confuse_matricx(y_test,y_predict):
  from sklearn.metrics import confusion_matrix
  import matplotlib.pyplot as plt
  from matplotlib.font_manager import FontProperties
  #y_test=np.concatenate(y_test)
  #y_predict=np.concatenate(y_predict)
  font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)
  confusion=confusion_matrix(y_predict,y_test)#产生混淆矩阵，结果更加清晰明了
  classes = list(set(y_test))#得到label的个数
  classes.sort()#将label按照顺序从小到大进行排列
  #绘图，显示混淆矩阵
  plt.imshow(confusion, cmap=plt.cm.Reds)
  indices = range(len(confusion))
  #设置x,y轴上面的数字
  plt.xticks(indices, classes)
  plt.yticks(indices, classes)
  #颜色bar,对应不同的深度
  plt.colorbar()#单个试图，多子图fig.colorbar()
  #定义坐标轴名称
  plt.xlabel("预测值",fontproperties=font)
  plt.ylabel("实际值",fontproperties=font)

  for first_index in range(len(confusion)):
      for second_index in range(len(confusion[first_index])):
           plt.text(first_index, second_index, confusion[first_index][second_index])

  plt.legend()
  plt.show()

if __name__ == "__main__":
  #file_name = r"C:\Users\PC\.spyder-py3\毕业设计\语谱图\tf_files\flower_photos\xuanwuyanzhu_v3\1001.jpg"
  model_file = r"F:/tensorflow/tf_files/inception_v3/graphV3.pb"
  label_file = r"F:/tensorflow/tf_files/inception_v3/labelsV3.txt"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  #inception-v3
  input_layer = "Mul"
  output_layer = "final_result"
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()
  path=r"F:\rock_image\训练集与测试集\test"
  total=0
  z_=0
  top2=0
  top3=0
  y_test=[]
  y_predict=[]
  for i in os.listdir(path):
      m=0
      t1=0
      t2=0
      n=0
      second_path=path+os.sep+i
      for j in os.listdir(second_path):
          lenth=len(os.listdir(second_path))
          n+=1
          file_name=second_path+os.sep+j
          if args.graph:
            model_file = args.graph
          if args.image:
            file_name = args.image
          if args.labels:
            label_file = args.labels
          if args.input_height:
            input_height = args.input_height
          if args.input_width:
            input_width = args.input_width
          if args.input_mean:
            input_mean = args.input_mean
          if args.input_std:
            input_std = args.input_std
          if args.input_layer:
            input_layer = args.input_layer
          if args.output_layer:
            output_layer = args.output_layer
        
          graph = load_graph(model_file)
          t = read_tensor_from_image_file(file_name,
                                          input_height=input_height,
                                          input_width=input_width,
                                          input_mean=input_mean,
                                          input_std=input_std)
        
          input_name = "import/" + input_layer
          output_name = "import/" + output_layer
          input_operation = graph.get_operation_by_name(input_name);
          output_operation = graph.get_operation_by_name(output_name);
        
          with tf.Session(graph=graph) as sess:
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                              {input_operation.outputs[0]: t})
            end=time.time()
          results = np.squeeze(results)
        
          top_k = results.argsort()[-5:][::-1]
          labels = load_labels(label_file)
          
          
          num=i
          y_test.append(num)
          y_predict.append(labels[top_k[0]])
          if(num==labels[top_k[0]]):
              m+=1
              t1+=1
              t2+=1
              print("第%d张图片识别正确，识别概率为%f"%(n,results[top_k[0]]))
          elif (i==labels[top_k[1]]):
              t1+=1
              t2+=1
              print("Top-2识别正确，top1为%s,识别概率为%f"%(labels[top_k[0]],results[top_k[0]]))
          elif (i==labels[top_k[2]]):
              t2+=1
              print("第%d张图片识别准确，结果为***********************************%s"%(n,labels[top_k[0]]))
          else:
              print("第%d张图片识别错误，错误结果为***********************************%s"%(n,labels[top_k[0]]))
      total+=lenth
      print("识别正确的张数为：",m)
      print("%s----的识别准确率为%f,Top2识别正确率为%f,Top3识别准确率%f"%(i,m/lenth,t1/lenth,t2/lenth))
      z_+=m
      top2+=t1
      top3+=t2
  print("整体识别准确率为%f,Top2识别准确率%f,Top3识别准确率为%f"%(z_/total,top2/total,top3/total))
  draw_confuse_matricx(y_test,y_predict)
'''       
print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
print(file_name)       
for i in top_k:
print(labels[i], results[i])
'''