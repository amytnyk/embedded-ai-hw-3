# Embedded AI HW3 Report

Report is written as an in-depth tutorial

## Converting YOLOv8 model

This was tested on **Ubuntu 22.04 (WSL)**

### ONNX

We will use already optimized ONNX model from **rknn_model_zoo**:
```shell
mkdir models
cd models
wget https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8n.onnx
```

Also, you can use **Ultralytics** to export yolov8n as int8 onnx model:
```shell
pip install ultralytics
yolo export model=models/yolov8n.pt format=onnx int8=True
```

### TFLite

Install latest tensorflow, tf-keras and onnx2tf:
```shell
pip install tensorflow tf-keras onnx2tf
```

Convert ONNX to TFLite with int8 quantization:
```shell
onnx2tf -i models/yolov8n.onnx -oiqt -o models/
```

### RKNN

#### Toolkit installation

In this case, you need to install RKNN toolkit from https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn-toolkit2

```shell
git clone https://github.com/rockchip-linux/rknn-toolkit2/
cd rknn-toolkit2

# Before executing change torch version from ==1.13.1 to >=1.3
pip install -r rknn-toolkit2/packages/requirements_cp{your_python_version}-1.6.0.txt
pip install rknn-toolkit2/packages/rknn_toolkit2-1.6.0+81f21f4d-cp{your_python_version}-cp{your_python_version}-linux_x86_64.whl
```

Note: rknn-toolkit is only available on x86_64

#### Converting ONNX to RKNN

```shell
python3 onnx2rknn.py
```

After this you should have `models/yolov8n.rknn` model

## Running YOLOv8 model on OrangePi 5+

### 
