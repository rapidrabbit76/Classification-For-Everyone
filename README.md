# Pytorch-lightning classification

Classification with pytorch lightning(as PL)

# **Requirements**

- [pip freeze](./requirements.txt)
- [conda env](./environment.yaml)

# Repository Tutorial

## Project Structure

```bash
RepoRootPath
├── models      # python module for training models
├── datamodules # python module for pl data module
├── transforms  # python module for data preprocessing
├── main.py     # Trainer
├── main.sh     # Training Recipe script
└── ...         # ETC ...
```

## Models Module Structure

```bash
models
├── LitBase                 # PL module base
│   └── lightning_model.py
├── Model_1                 # Model 1
│   ├── blocks.py           # Models sub blocks
│   ├── models.py           # Pure pytorch model define
│   └── lightning_model.py  # Loss and optimizer setting using PL
├── Model_2
├── Model_N
...
```

### LitBase

```python
# models.LitBase.lightning_model.py
class LitBase(pl.LightningModule, metaclass=ABCMeta):
    @abstractmethod
    def configure_optimizers(self):
        return super().configure_optimizers()
    """
    def initialize_weights ...
    def forward ...
    def training_step ...
    def validation_step ...
    def test_step ...
    def _validation_test_common_epoch_end ...
    def validation_epoch_end ...
    def test_epoch_end ...
    """
```

### Implemented Models

```python
# models.LeNet5.lightning_model.py
class LitLeNet5(LitBase):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = LeNet5(
            image_channels=self.hparams.image_channels,
            num_classes=self.hparams.num_classes,
        )
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
```

# Install

## Install from source code

### using anaconda/miniconda

```bash
$ conda env create --file environment.yaml
```

### using pip

```bash
$ pip install -r requirements.txt
```

## Install using docker/docker-compose

```bash
$ export USERID=$(id -u)
$ export GROUPID=$(id -g)
$ docker-compose up -d
```

```yaml
version: "3.7"
    trainer:
    build: .
    user: "${USERID}:${GROUPID}"
    volumes:
        - .:/training
        - /{YOUR_DATA_SET_DIR_PATH}:/DATASET # !!Setting dataset path!!
    command: tail -f /dev/null
```

# Training

Please see the ["Recipes"](./md/Recipes.md)

# Experiment results

Please see the ["Experiment results"](./md/Experiment.md)

# Supported model architectures

Please see the ["Supported Model"](./md/Supported%20Model.md)

# Supported dataset

Please see the ["Supported Dataset"](./md/Supported%20Dataset.md)
