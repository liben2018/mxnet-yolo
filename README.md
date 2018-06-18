# YOLO-v2: Real-Time Object Detection

## History of mAP (on VOC2007)

### Resnet-50: 
#### Exp1 
- lr=0.001, steps=(90, 180), data-shape=416. However, adding the first 80 epoch, in fact the steps = (90 + 80, 180 + 80)
- first: just runing in 80 epoch then stop because of batch-size=30!
```
python train-416.py --gpus 0,1 --network resnet50_yolo --data-shape 416 --pretrained model/resnet-50 --epoch 0 --log train_416.log --min-random-shape 320 --batch-size 30
```
- second: continuing to train based on the 80-th weight in the first, meanwhile tuning batch-size to 28!
```
python train-416.py --gpus 0,1 --network resnet50_yolo --data-shape 416 --pretrained model/yolo2_resnet50_416 --epoch 80 --log train_416_80.log --min-random-shape 320 --batch-size 28
```
- 2018/06/15: Epoch[236] Validation-mAP=0.759613
- 2018/06/14: Epoch[174] Validation-mAP=0.759229
- 2018/06/14: Epoch[126] Validation-mAP=0.757837
- 2018/06/14: Epoch[105] Validation-mAP=0.757725
- 2018/06/14: Epoch[104] Validation-mAP=0.757814
- 74 mAP by original repo (https://github.com/zhreshold/mxnet-yolo)

#### Exp2 
- tuning random-shape-epoch from 10 to 1, other almost same with original repo.
- Command: 
```
python train-416.py --network resnet50_yolo --batch-size 28 --pretrained model/resnet-50 --epoch 0 --gpus 0,1 --begin-epoch 0 --end-epoch 240 --data-shape 416 --random-shape-epoch 1 --min-random-shape 320 --max-random-shape 608 --lr 0.001 --lr-steps 90,180 --lr-factor 0.1 --log “train-exp2.log” --num-class 20 --num-example 16551 --nms 0.45 --overlap 0.5
```
- 2018/06/15: Epoch[236] Validation-mAP=0.707528

#### Exp3
- lr-steps 180,360
- Command:
```
python train-416.py --network resnet50_yolo --batch-size 28 --pretrained model/resnet-50 --epoch 0 --gpus 0,1 --begin-epoch 0 --end-epoch 540 --data-shape 416 --random-shape-epoch 10 --min-random-shape 320 --max-random-shape 608 --lr 0.001 --lr-steps 180,360 --lr-factor 0.1 --log train-exp3.log --num-class 20 --num-example 16551 --nms 0.45 --overlap 0.5
```
- 2018/06/18: Epoch[256] Validation-mAP=0.758914

#### Exp4
- lr-factor=0.5，lr-steps=[90，180，270，360，450]，lr=[0.001，0.0005，0.00025，0.000125，0.0000625，0.00003125]
- Command:
```
python train-416.py --network resnet50_yolo --batch-size 28 --pretrained model/resnet-50 --epoch 0 --gpus 0,1 --begin-epoch 0 --end-epoch 540 --data-shape 416 --random-shape-epoch 10 --min-random-shape 320 --max-random-shape 608 --lr 0.001 --lr-steps 90,180,270,360,450 --lr-factor 0.5 --log train-exp4.log --num-class 20 --num-example 16551 --nms 0.45 --overlap 0.5
```

### Darkent19:
- 71 mAP by original repo (https://github.com/zhreshold/mxnet-yolo)

## Difference between official Mxnet and the one used in the repo

added the following operators:
- yolo_output.* to src/operator/contrib
- stack_neighbor into src/operator/matrix_op

https://github.com/zhreshold/mxnet-yolo/issues/7 

## Disclaimer
Re-implementation of original yolo-v2 which is based on [darknet](https://github.com/pjreddie/darknet).

The arXiv paper is available [here](https://arxiv.org/pdf/1612.08242.pdf).

## Demo

![demo1](https://user-images.githubusercontent.com/3307514/28980832-29bb0262-7904-11e7-83e3-a5fec65e0c70.png)

## Getting started
- Build from source, this is required because this example is not merged, some
custom operators are not presented in official MXNet. [Instructions](http://mxnet.io/get_started/install.html)
- Install required packages: `cv2`, `matplotlib`

## Try the demo
- Download the pretrained [model](https://github.com/zhreshold/mxnet-yolo/releases/download/0.1-alpha/yolo2_darknet19_416_pascalvoc0712_trainval.zip)(darknet as backbone), or this [model](https://github.com/zhreshold/mxnet-yolo/releases/download/v0.2.0/yolo2_resnet50_voc0712_trainval.tar.gz)(resnet50 as backbone) and extract to `model/` directory.
- Run
```
# cd /path/to/mxnet-yolo
python demo.py --cpu
# available options
python demo.py -h
```

## Train the model
- Grab a pretrained model, e.g. [`darknet19`](https://github.com/zhreshold/mxnet-yolo/releases/download/0.1-alpha/darknet19_416_ILSVRC2012.zip)
- (optional) Grab a pretrained resnet50 model, [`resnet-50-0000.params`](http://data.dmlc.ml/models/imagenet/resnet/50-layers/resnet-50-0000.params),[`resnet-50-symbol.json`](http://data.dmlc.ml/models/imagenet/resnet/50-layers/resnet-50-symbol.json), this will produce slightly better mAP than `darknet` in my experiments.
- Download PASCAL VOC dataset.
```
cd /path/to/where_you_store_datasets/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
ln -s /path/to/VOCdevkit /path/to/mxnet-yolo/data/VOCdevkit
```
- Create packed binary file for faster training
```
# cd /path/to/mxnet-ssd
bash tools/prepare_pascal.sh
# or if you are using windows
python tools/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target ./data/train.lst
python tools/prepare_dataset.py --dataset pascal --year 2007 --set test --target ./data/val.lst --shuffle False
```
- Start training
```
python train.py --gpus 0,1,2,3 --epoch 0
# choose different networks, such as resnet50_yolo
python train.py --gpus 0,1,2,3 --network resnet50_yolo --data-shape 416 --pretrained model/resnet-50 --epoch 0
# see advanced arguments for training
python train.py -h
```
