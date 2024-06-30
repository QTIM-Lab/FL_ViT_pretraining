# Set up SSL-encrypted gRPC Connection on AWS EC2
This file contains instructions on how to run an SSL-encrpyted gRPC server on EC2, and have clients to connect to it to do federated learning.

## Environment Setup

Install `git` and `conda`.

```bash
sudo yum update -y
sudo yum install git -y
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
conda activate flower
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/advanced-tensorflow . && rm -rf flower && cd advanced-tensorflow
pip install -r requirements.txt # Comment out the second line
pip install "flwr-datasets[vision]"
```

## Generate necessary certificates and keys 
Modify `certificats/certificate.conf` file's `[alt_names]` fields by changing `DNS.1=<your_ec2_dns>` and `IP.1=<your_ec2_ip>`. Copy the contents in `.cache/certificates/ca.crt` to your local computer.

## Start the server

Open port 8080 for inbound connection and start the server by `python server.py`.

## Test connection to server on your local computer

On you **local computer**, also clone the flower example and do necessary installation.

```bash
conda create -n flower python=3.10 --y
conda activate flower
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/advanced-tensorflow . && rm -rf flower && cd advanced-tensorflow
pip install -r requirements.txt # Comment out the second line
pip install "flwr-datasets[vision]"
```

Go to `client.py` and modify the `server_address` and `root_certificates` accordingly. 

```python
    fl.client.start_client(
        server_address="3.88.5.129:8080",
        client=client,
        root_certificates=Path("./ca.crt").read_bytes(),
    )
```

Create a shell script (e.g., `test.sh`) with the following content.

```bash
#!/bin/bash

for i in $(seq 0 9); do
    echo "Starting client $i"
    python client.py --client-id=${i} --toy &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
```


Run the shell script to connect to the server.

```bash
chmod +x test.sh # you only need to do it once
./test.sh
```