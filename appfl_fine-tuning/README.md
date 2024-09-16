## Finetune a Vision Transformer Model Using APPFL

**There are the detailed instructions on fine-tuning ViT using APPFL at [here](https://appfl.ai/en/latest/tutorials/examples_vit_finetuning.html)**

**There are video tutorials available on [YouTube](https://youtu.be/m4rdOub2Y_o)**

### Installation

We first create a `conda` virtual environment called `appfl`, and then install `mpi4py` via `conda` (`mpi4py` is a required package to install `appfl` latter, but it cannot be installed via `pip` on AWS EC2 instance, so we need to install it first).

```bash
conda create -n appfl python=3.10 --y
conda activate appfl
conda install mpi4py
```

Then, we can install the `appfl` package, there are two different ways: (1) Install directly via `pip`; (2) Build `appfl` from its source code.

```bash
# Install directly via pip
pip install "appfl[examples]"
# OR
# Install via building from source code
git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
cd APPFL
pip install -e ".[examples]"
```

### Code Structure

Below shows the structure for the `appfl_vit_finetuning_server` and `appfl_vit_finetuning_client` directories. 

```
appfl_vit_finetuning_server
    ├── resources
    │   ├── vit.py                  # Model architecture
    │   ├── metric.py               # Metric function
    │   └── vit_ft_trainer.py       # Trainer
    ├── config.yaml                 # Server configuration file
    └── run_server.py               # Server launching script

appfl_vit_finetuning_client
    ├── resources
    │   └── vit_fake_dataset.py      # Dataset loader
    ├── config.yaml                  # Client configuration file
    └── run_client.py                # Client launching script
```

### Launch Experiments

On the server-side,

```bash
python run_server.py
```

On the client-side,

```bash
python run_client.py
```