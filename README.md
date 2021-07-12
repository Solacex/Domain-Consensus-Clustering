
# [CVPR2021] Domain Consensus Clustering for Universal Domain Adaptation

[[Paper]](http://reler.net/papers/guangrui_cvpr2021.pdf)


### Prerequisites

To install requirements:

```setup
pip install -r requirements.txt
```

- Python 3.6
- GPU Memory: 10GB
- Pytorch 1.4.0



## Getting Started

Download the dataset: Office-31, OfficeHome, VisDA, DomainNet. 

Data Folder structure: 
```
Your dataset DIR:
|-Office/domain_adaptation_images
| |-amazon
| |-webcam
| |-dslr
|-OfficeHome
| |-Art
| |-Product
| |-...
|-VisDA
| |-train
| |-validataion
|-DomainNet
| |-clipart
| |-painting
| |-...
```
You need you modify the data_path in config files, i.e., config.root

## Training

Train on one transfer of Office: 
```
CUDA_VISIBLE_DEVICES=0 python office_run.py note=EXP_NAME setting=uda/osda/pda source=amazon target=dslr
```

To train on six transfers of Office:
```
CUDA_VISIBLE_DEVICES=0 python office_run.py note=EXP_NAME setting=uda/osda/pda transfer_all=1
```



Train on OfficeHome: 
```
CUDA_VISIBLE_DEVICES=0 python officehome_run.py note=EXP_NAME setting=uda/osda/pda source=Art target=Product
```
or 
```
CUDA_VISIBLE_DEVICES=0 python officehome_run.py note=EXP_NAME setting=uda/osda/pda transfer_all=1 
```

The final results (including the best and the last) will be saved in the ./snapshot/EXP_NAME/result.txt. 

Notably, transfer_all wil consumes more shared memory. 


## Citation 

If you find it helpful, please consider citing: 

```
@inproceedings{li2021DCC,
  title={Domain Consensus Clustering for Universal Domain Adaptation},
  author={Li, Guangrui and Kang, Guoliang and Zhu, Yi and Wei, Yunchao and Yang, Yi},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}

```

