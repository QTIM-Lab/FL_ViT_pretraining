# Set up SSL-encrypted gRPC Connection on AWS EC2
This file contains instructions on how to run an SSL-encrpyted gRPC server on EC2, and have clients to connect to it to do federated learning.

## Environment Setup

Install `conda`.

```bash
mkdir conda
cd conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
cd
```

Create a virtual environment, and do necessary installation.

```bash
conda create -n flower python=3.10 --y
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/advanced-tensorflow . && rm -rf flower && cd advanced-tensorflow
pip install -r requirements.txt # Comment out the second line
pip install "flwr-datasets[vision]"
```

## Generate necessary certificates and keys 
Modify `certificats/certificate.conf` file's `[alt_names]` fields by changing `DNS.1=<your_ec2_dns>` and `IP.1=<your_ec2_ip>`.