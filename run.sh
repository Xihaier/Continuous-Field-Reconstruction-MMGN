python main.py datamodule.model_name=ResMLP datamodule.task=task1 datamodule.sampling_rate=[0.01,0.05] model=ResMLP
python main.py datamodule.model_name=FFN_P  datamodule.task=task1 datamodule.sampling_rate=[0.01,0.05] model=FFN_P
python main.py datamodule.model_name=FFN_G  datamodule.task=task1 datamodule.sampling_rate=[0.01,0.05] model=FFN_G
python main.py datamodule.model_name=SIREN  datamodule.task=task1 datamodule.sampling_rate=[0.01,0.05] model=SirenNet
python main.py datamodule.model_name=MMGN   datamodule.task=task1 datamodule.sampling_rate=[0.01,0.05] model=MMGNet