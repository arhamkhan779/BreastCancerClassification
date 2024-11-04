from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngstionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir:Path
    base_model_path:Path
    updated_base_model_path:Path
    params_img_size:list
    params_learning_rate:float
    params_include_top:bool
    params_weight:str
    params_classes:int


@dataclass(frozen=True)
class TrainingConfig:
    results: Path
    root_dir:Path
    trained_model_path: Path
    update_base_model_path: Path
    training_data: Path
    params_epoch: int
    params_batch_size: int
    params_is_augmentation: bool
    params_img_size: list