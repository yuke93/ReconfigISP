## ReconfigISP [[arXiv](https://arxiv.org/abs/2109.04760)][[Project Page](https://www.mmlab-ntu.com/project/reconfigisp/)]
This repo contains the main code of ReconfigISP.

### OnePlus Dataset
The OnePlus dataset is available [here](https://drive.google.com/file/d/1Dw0btNcBHN0diaATEFiE_8PERNvniMHU/view?usp=sharing).

### Train
training without proxy tuning
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM train.py --opt options/train/{YOUR_OPTION_FILE}.yml --launcher pytorch
```

training with proxy tuning
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM train_ft.py --opt options/train/{YOUR_OPTION_FILE}.yml --launcher pytorch
```

### Test
normal inference
```
python test.py --opt options/test/{YOUR_OPTION_FILE}.yml
```

split into patches for inference (save GPU memory)
```
python test_split.py --opt options/test/{YOUR_OPTION_FILE}.yml
```

split into patches to test object detection
```
python test_yolo_split.py --opt options/test/{YOUR_OPTION_FILE}.yml
```

### Acknowledgement
The implementation of architecture search is borrowed from [DARTS](https://github.com/quark0/darts)

The images are annotated by [labelme](https://github.com/wkentaro/labelme).