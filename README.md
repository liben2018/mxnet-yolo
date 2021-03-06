# YOLO-v2: Real-Time Object Detection

## Todo list
- Training different backbone from scratch
- Get new network weights by training on imagenet directly or transfer learning
- Loss function
- Adding --num_layers in train.py for choosing different network

## History of mAP (on VOC2007)

### Resnet-50: 
#### Exp1 
- lr=0.001, steps=(90, 180), data-shape=416. However, adding the first 80 epoch, in fact the steps = (90 + 80, 180 + 80)
- first: just runing in 80 epoch then stop because of batch-size=30!
```
python train-416.py --gpus 0,1 --network resnet50_yolo --data-shape 416 --pretrained model/resnet-50 \ 
                    --epoch 0 --log train_416.log --min-random-shape 320 --batch-size 30
```
- second: continuing to train based on the 80-th weight in the first, meanwhile tuning batch-size to 28!
```
python train-416.py --gpus 0,1 --network resnet50_yolo --data-shape 416 --pretrained model/yolo2_resnet50_416 \
                    --epoch 80 --log train_416_80.log --min-random-shape 320 --batch-size 28
```
- 2018/06/15: Epoch[236] Validation-mAP=0.759613
- 2018/06/14: Epoch[174] Validation-mAP=0.759229
- 2018/06/14: Epoch[126] Validation-mAP=0.757837
- 2018/06/14: Epoch[105] Validation-mAP=0.757725
- 2018/06/14: Epoch[104] Validation-mAP=0.757814
- 74 mAP by original repo (https://github.com/zhreshold/mxnet-yolo)
- 76.8 mAP at (416 x 416) by original paper.

#### Exp2 
- tuning random-shape-epoch from 10 to 1, other almost same with original repo.
- Command: 
```
python train-416.py --network resnet50_yolo --batch-size 28 --pretrained model/resnet-50 \
                    --epoch 0 --gpus 0,1 --begin-epoch 0 --end-epoch 240 --data-shape 416 \
                    --random-shape-epoch 1 --min-random-shape 320 --max-random-shape 608 \
                    --lr 0.001 --lr-steps 90,180 --lr-factor 0.1 --log train-exp2.log \
                    --num-class 20 --num-example 16551 --nms 0.45 --overlap 0.5
```
- 2018/06/15: Epoch[236] Validation-mAP=0.707528

#### Exp3
- lr-steps 180,360
- Command:
```
python train-416.py --network resnet50_yolo --batch-size 28 --pretrained model/resnet-50 \
                    --epoch 0 --gpus 0,1 --begin-epoch 0 --end-epoch 540 --data-shape 416 \
                    --random-shape-epoch 10 --min-random-shape 320 --max-random-shape 608 \
                    --lr 0.001 --lr-steps 180,360 --lr-factor 0.1 --log train-exp3.log \
                    --num-class 20 --num-example 16551 --nms 0.45 --overlap 0.5
```
- 2018/06/18: Epoch[256] Validation-mAP=0.758914

#### Exp4
- lr-factor=0.5，lr-steps=[90，180，270，360，450]，lr=[0.001，0.0005，0.00025，0.000125，0.0000625，0.00003125]
- Command:
```
python train-416.py --network resnet50_yolo --batch-size 28 --pretrained model/resnet-50 \
                    --epoch 0 --gpus 0,1 --begin-epoch 0 --end-epoch 540 --data-shape 416 \
                    --random-shape-epoch 10 --min-random-shape 320 --max-random-shape 608 \
                    --lr 0.001 --lr-steps 90,180,270,360,450 --lr-factor 0.5 --log train-exp4.log \
                    --num-class 20 --num-example 16551 --nms 0.45 --overlap 0.5

python train-416.py --network resnet50_yolo --batch-size 28 --pretrained model/resnet-50 \
                    --epoch 0 --gpus 0,1 --begin-epoch 0 --end-epoch 540 --data-shape 416 \
                    --random-shape-epoch 10 --min-random-shape 320 --max-random-shape 608 \
                    --lr 0.001 --lr-steps 90,180,270,360,450 --lr-factor 0.5 --log train-exp4.log \
                    --num-class 20 --num-example 16551 --nms 0.45 --overlap 0.5 --resume 288
```
- 2018/06/21: Epoch[477] Validation-mAP=0.759190
- 2018/06/20: Epoch[408] Validation-mAP=0.759149
- 2018/06/19: Epoch[276] Validation-mAP=0.757242
- 2018/06/19: Epoch[233] Validation-mAP=0.755262

#### Exp5
- lr-factor=0.1，lr-steps=[90，180，270，360，450]，lr=[0.001，0.0001，0.00001，0.000001，0.0000001，0.00000001]
- Command:
```
python train-416.py --network resnet50_yolo --batch-size 28 --pretrained model/resnet-50 \
                    --epoch 0 --gpus 0,1 --begin-epoch 0 --end-epoch 540 --data-shape 416 \
                    --random-shape-epoch 10 --min-random-shape 320 --max-random-shape 608 \
                    --lr 0.001 --lr-steps 90,180,270,360,450 --lr-factor 0.1 --log train-exp5.log
```
- 2018/06/22: 

#### Exp6
- lr-factor=0.1，lr-steps=[90，180]，lr=[0.001，0.0001，0.00001]
- Command:
```
python train-416-resnet152.py --network resnet152_yolo --batch-size 28 --pretrained scratch\
                    --epoch 0 --gpus 0,1 --begin-epoch 0 --end-epoch 540 --data-shape 416 \
                    --random-shape-epoch 10 --min-random-shape 320 --max-random-shape 608 \
                    --lr 0.001 --lr-steps 90,180 --lr-factor 0.1 --log train-exp6.log
```
- 2018/06/22: 
#### Exp7
- data-shape != 416
- The results are not good becouse the pretrained models are trained for data-shape 416?

### Darkent19:
- 71 mAP by original repo (https://github.com/zhreshold/mxnet-yolo)

## Difference between official Mxnet and the one used in the repo

added the following operators:
- yolo_output.* to src/operator/contrib

https://github.com/zhreshold/mxnet-yolo/issues/7 

## Changing network
- create file: mxnet-yolov2/symbol/symbol_resnet152_yolo.py 
- get weights file for resnet152
- change num_layers (from 50 to 152) in symbol_resnet152_yolo.py:
```
def get_symbol(num_classes=20, nms_thresh=0.5, force_nms=False, **kwargs):
    #body = resnet.get_symbol(num_classes, 50, '3,224,224')
    body = resnet.get_symbol(num_classes, 152, '3,224,224')
```
- using resnet152_yolo as --network in train.py
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
# choose different networks, such as resnet152_yolo
python train.py --gpus 0,1,2,3 --network resnet152_yolo --data-shape 416 --pretrained model/resnet-50 --epoch 0
# see advanced arguments for training
python train.py -h
```
