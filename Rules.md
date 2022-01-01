# Repository rules
- 코드 스타일은 Black을 유지
- DataLoader는 [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html)를 적극 활용
- logging은 [Weights & Biases](https://wandb.ai/site)를 사용
- 학습 완료 시 모델 디렉토리 내부에 사용한 hyperparameters 작성
- 학습 완료 시 모델 디렉토리 내부에 성능평가 결과 작성 
- 커스템 데이터 셋을 사용할 경우 출처를 표기 (공개 가능 한 경우)