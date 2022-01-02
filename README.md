# **Requirements**

- torch==1.10.0
- pytorch-lightning==1.5.7
- wandb==0.12.9
- albumentations==0.16.1

# **Rules**

- [Repository Rules](./Rules.md)

# Repository Tutorial

## 1. Code Structure

각 구현체의 경우 디렉터리로 구분되어 있으며 각 구현체 디렉터리의 경우 아래와 같은 구조를 기본으로 합니다.

별도의 구성요소가 포함되면 각 구현체 README.md에 설명을 포함합니다.

```bash
# [*]       : 학습에 꼭 필요 혹은 기본 구성요소
RepoRootPath
├── LeNet-5
│   ├── config.yml          # Hyperparameters[*]
│   ├── datamodule.py       # Pytorch-Lightning Data Module[*]
│   ├── model.py            # Model[*]
│   ├── sweep.yaml          # Sweep config for hyperparameter tuning
│   ├── train.py            # Pytorch-Lightning LightningModule & train script
├── utils.py                # Utility script
├── environment.yaml        # conda Env
├── requirements.txt        # pip freeze
├── ... Etcs
```

## 2. train

**해당 레포는 [Pytorch-Lightning](https://www.pytorchlightning.ai/)과 [Weights & Biases](https://wandb.ai/)를 사용합니다.**

LightningModule에 관한 자세한 내용은 [링크](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=LightningModule)를 참고

```python
class LitModel(pl.LightningModule):
    def __init__( self, ...):
        """
            - hyperparameter save
            - build model
            - define loss
            - define metrics
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = MyModel(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ model forward """
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        ...

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        ...

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        ...

    def configure_optimizers(self):
        ...
```

## 3. Run Model

```python

def train():
    # Hyperparameters
    config = utils.get_config()
    hparams = config.hparams
    seed_everything(config.seed)

    # Dataloader
    dm = datamodule.QMnistDataModule(config.data_dir)

    # Model
    model = MyModel(...)

    # Logger
    wandb_logger = WandbLogger(
        name=f"{config.project_name}-{config.dataset}",
        project=config.project_name,
        save_dir=config.save_dir,
        log_model="all",
    )
    wandb_logger.experiment.config.update(hparams)
    wandb_logger.watch(lenet5, log="all", log_freq=100)

    # Trainer setting
    callbacks = [ ... ]

    trainer: pl.Trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=hparams.epochs,
        callbacks=callbacks,
        ...
    )

    # Train
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    # Finish
    wandb_logger.experiment.unwatch(lenet5)

    # Model to Torchscript
    ...

if __name__ == "__main__":
    train()
```
