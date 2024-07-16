# Generalized Continual Category Discovery (GCCD)
### Setup:
```bash
pip install requirements.txt
```

### To run pretrained ViT download a model from https://github.com/facebookresearch/dino: 
```bash
mkdir pretrained && cd pretrained
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
```

### Download datasets
### CUB200
```bash
cd data
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
tar zxvf 'CUB_200_2011.tgz?download=1'
```
### DomainNet
```bash
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
```

### Run CAMP method on 5 datasets:
```bash
bash scripts/camp.sh
```

### Run GCD method on 5 datasets:
```bash
bash scripts/gcd.sh
```

### Run GCD+FD method on 5 datasets:
```bash
bash scripts/gcd_fd.sh
```

### Run GCD+EWC method on 5 datasets:
```bash
bash scripts/gcd_ewc.sh
```

### Run Proxy Anchors method on 5 datasets:
```bash
bash scripts/proxy_anchors.sh
```

If you find this code or paper useful we will appreciate if you cite us:
```commandline
@article{rypesc2024category,
  title={Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery},
  author={Rype{\'s}{\'c}, Grzegorz, and Marczak, Daniel and Cygert, Sebastian and Trzci{\'n}ski, Tomasz and Twardowski, Bart{\l}omiej},
  journal={arXiv preprint arXiv:2308.12112},
  year={2024}
}
```