# yolov7-education-tensorrt
A python and c++ implementation of yolov7, head , rise hand and stand detection.


# Python Environment Requirement 
你可以将下面内容copy到txt中，然后运行
```shell
conda create -n yolo python==3.8
conda activate yolo
pip install -r requirement.txt
```

```shell
absl-py==1.4.0
appdirs==1.4.4
asttokens==2.2.1
backcall==0.2.0
cachetools==5.3.0
certifi @ file:///croot/certifi_1671487769961/work/certifi
charset-normalizer==3.1.0
cmake==3.26.1
coloredlogs==15.0.1
contourpy==1.0.7
cycler==0.11.0
decorator==5.1.1
executing==1.2.0
filelock==3.11.0
flatbuffers==23.3.3
fonttools==4.39.3
google-auth==2.17.2
google-auth-oauthlib==1.0.0
grpcio==1.53.0
humanfriendly==10.0
idna==3.4
importlib-metadata==6.1.0
importlib-resources==5.12.0
ipython==8.12.0
jedi==0.18.2
Jinja2==3.1.2
kiwisolver==1.4.4
lit==16.0.0
Mako==1.2.4
Markdown==3.4.3
markdown-it-py==2.2.0
MarkupSafe==2.1.2
matplotlib==3.7.1
matplotlib-inline==0.1.6
mdurl==0.1.2
mpmath==1.3.0
networkx==3.1
numpy==1.23.5
nvidia-cublas-cu11==11.10.3.66
nvidia-cublas-cu12==12.1.0.26
nvidia-cuda-cupti-cu11==11.7.101
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cuda-runtime-cu12==12.1.55
nvidia-cudnn-cu11==8.5.0.96
nvidia-cudnn-cu12==8.9.0.131
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.2.10.91
nvidia-cusolver-cu11==11.4.0.1
nvidia-cusparse-cu11==11.7.4.91
nvidia-nccl-cu11==2.14.3
nvidia-nvtx-cu11==11.7.91
oauthlib==3.2.2
onnx==1.13.1
onnx-simplifier==0.4.19
onnxruntime==1.14.1
opencv-python==4.7.0.72
packaging==23.0
pandas==2.0.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==9.5.0
platformdirs==3.2.0
prompt-toolkit==3.0.38
protobuf==3.20.3
psutil==5.9.4
ptyprocess==0.7.0
pure-eval==0.2.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycuda==2020.1
Pygments==2.14.0
pyparsing==3.0.9
python-dateutil==2.8.2
pytools==2022.1.14
pytz==2023.3
PyYAML==6.0
requests==2.28.2
requests-oauthlib==1.3.1
rich==13.3.3
rsa==4.9
scipy==1.10.1
seaborn==0.12.2
six==1.16.0
stack-data==0.6.2
sympy==1.11.1
tensorboard==2.12.1
tensorboard-data-server==0.7.0
tensorboard-plugin-wit==1.8.1
tensorrt==8.6.0
thop==0.1.1.post2209072238
torch==2.0.0
torchvision==0.15.1
tqdm==4.65.0
traitlets==5.9.0
triton==2.0.0
typing_extensions==4.5.0
tzdata==2023.3
urllib3==1.26.15
wcwidth==0.2.6
Werkzeug==2.2.3
zipp==3.15.0
```
# Custom .pt Model to TensorRT Engine Model
将yolo转为trt模型有两个选择，1是.pt->wts->engine,2是.pt->onnx->engine,这里选择第二种.
## Custom .pt Model to onnx
这里使用`yoloV7`自带的export.py文件，将训练好的.pt文件导出为onnx。

```
python export.py --weights /data/tbw_data/hand/yolov7/runs/train/yolov714/weights/best.pt  --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
```
这里 `--end2end`参数即选择将nms纳入到模型当中，生成end2end的模型,`--topk-all`参数为最多检测的最后结果为多少。
运行成功后，会在.pt对应路径下生成onnx模型。
```
-rw-rw-r--. 1 tbw tbw 140M 4月  13 11:21 best.onnx
-rw-rw-r--. 1 tbw tbw 285M 4月  10 14:09 best.pt
-rw-rw-r--. 1 tbw tbw 140M 4月  13 11:21 best.torchscript.pt
-rw-rw-r--. 1 tbw tbw 140M 4月  13 11:21 best.torchscript.ptl
```

## Custom onnx Model to TensorRT Model
这里用两个方法将onnx模型生成TensorRT模型，分别是使用`trtexec`来直接导出或者[Linaom1214](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)中的[export.py](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/export.py)

1. trtexec

```shell
/usr/src/tensorrt/bin/trtexec --onnx=yolov7.onnx --saveEngine=yolov7-nms.trt --fp16 --workspace=64
```
其中参数`--onnx`输入的是Custom Onnx Model的具体路径，而`--saveEngine`输入的是导出的TensorRT模型想保存的具体路径。参数`--workspace`设置工作空间大小单位是MB(默认为16MB)，为了避免出现空间不够的情况，可以酌情调大一点。
```
Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
```

使用`trtexec --help`命令可以查看一些参数，这里贴一些参数
```shell
# 生成静态batchsize的engine
./trtexec 	--onnx=<onnx_file> \ 						#指定onnx模型文件
        	--explicitBatch \ 							#在构建引擎时使用显式批大小(默认=隐式)显示批处理
        	--saveEngine=<tensorRT_engine_file> \ 		#输出engine
        	--workspace=<size_in_megabytes> \ 			#设置工作空间大小单位是MB(默认为16MB)
        	--fp16 										#除了fp32之外，还启用fp16精度(默认=禁用)

# 生成动态batchsize的engine
./trtexec 	--onnx=<onnx_file> \						#指定onnx模型文件
        	--minShapes=input:<shape_of_min_batch> \ 	#最小的NCHW
        	--optShapes=input:<shape_of_opt_batch> \  	#最佳输入维度，跟maxShapes一样就好
        	--maxShapes=input:<shape_of_max_batch> \ 	#最大输入维度
        	--workspace=<size_in_megabytes> \ 			#设置工作空间大小单位是MB(默认为16MB)
        	--saveEngine=<engine_file> \   				#输出engine
        	--fp16   									#除了fp32之外，还启用fp16精度(默认=禁用)


# 小尺寸的图片可以多batchsize即8x3x416x416
/home/zxl/TensorRT-7.2.3.4/bin/trtexec  --onnx=yolov4_-1_3_416_416_dynamic.onnx \
                                        --minShapes=input:1x3x416x416 \
                                        --optShapes=input:8x3x416x416 \
                                        --maxShapes=input:8x3x416x416 \
                                        --workspace=4096 \
                                        --saveEngine=yolov4_-1_3_416_416_dynamic_b8_fp16.engine \
                                        --fp16

# 由于内存不够了所以改成4x3x608x608
/home/zxl/TensorRT-7.2.3.4/bin/trtexec  --onnx=yolov4_-1_3_608_608_dynamic.onnx \
                                        --minShapes=input:1x3x608x608 \
                                        --optShapes=input:4x3x608x608 \
                                        --maxShapes=input:4x3x608x608 \
                                        --workspace=4096 \
                                        --saveEngine=yolov4_-1_3_608_608_dynamic_b4_fp16.engine \
                                        --fp16
```

想要测试运行一下导出的trt模型，可以使用python或者c++代码，建议运行:

```
git clone https://github.com/Linaom1214/tensorrt-python.git
cd tensorrt-python/cpp/end2end
```
根据自己的环境修改CMakeLists.txt，这是63nx修改的内容，主要是为了找到Opencv环境

```CMake
cmake_minimum_required(VERSION 2.6)

project(yolo)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
# add_definitions("-Wall -g")
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# find_package(OpenCV)
find_package(
 OpenCV 
 REQUIRED
  # PATHS /home/tthd/project/opencv/opencv4.6.0/build
 PATHS /data/project/opencv/opencv4.6.0/build
  NO_DEFAULT_PATH
)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolo ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(yolo nvinfer)
target_link_libraries(yolo nvinfer_plugin)
target_link_libraries(yolo cudart)

target_link_libraries(yolo ${OpenCV_LIBS})

add_definitions(-O2 -pthread)
```
同时需要修改`main.cpp`中的内容,即`Yolo::Yolo(char* model_path)`函数中`engine->getBindingIndex`的key，因为这里不是用`tensorrt-python`的`export.py`导出的，输入和输出的参数名称上会有差别，具体名称请参考[yolov7/export.py](https://github.com/WongKinYiu/yolov7/blob/3b41c2cc709628a8c1966931e696b14c11d6db0c/export.py#L159)中的设置。具体修改如下:

```cpp
  auto out_dims1 = engine->getBindingDimensions(engine->getBindingIndex("num_dets"));
  out_size1 = 1;
  for (int j = 0; j < out_dims1.nbDims; j++) {
    out_size1 *= out_dims1.d[j];
  }
  auto out_dims2 = engine->getBindingDimensions(engine->getBindingIndex("det_boxes"));
  out_size2 = 1;
  for (int j = 0; j < out_dims2.nbDims; j++) {
    out_size2 *= out_dims2.d[j];
  }
  auto out_dims3 = engine->getBindingDimensions(engine->getBindingIndex("det_scores"));
  out_size3 = 1;
  for (int j = 0; j < out_dims3.nbDims; j++) {
    out_size3 *= out_dims3.d[j];
  }
  auto out_dims4 = engine->getBindingDimensions(engine->getBindingIndex("det_classes"));
```

随后在`tensorrt-python/cpp/end2end`路径下运行

```shell
cd end2end && mkdir build && build
cmake ..
make
./yolo -model_path  engine   -image_path xxx.jpg
```
即在build下生成了result.jpg结果

2. Python & export.py

对于使用python开发的项目，即可运行下面命令进行测试，注意，下面代码并没有在本人的nx环境中进行测试。
```shell
git clone https://github.com/Linaom1214/tensorrt-python.git
cd tensorrt-python
python3 ./export.py -o {yolov7.onnx} -e {yolov7-nms.trt} -p fp16 
python3 trt.py -e yolov7-nms.trt  -i src/1.jpg -o yolov7.jpg --end2end 
```
