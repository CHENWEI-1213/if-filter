import sys
# sys.path.insert(0, '../../..')
from trainer_line_ctc import TrainerLineCTC
from models_line_ctc import Decoder
from basic.models import FCN_Encoder
from torch.optim import Adam
from basic.generic_dataset_manager import OCRDataset
from basic.generic_dataset_manager import DatasetManager
import torch.multiprocessing as mp
import torch
import skimage
import os
from trak import TRAKer
from argparse import ArgumentParser
from trak.gradient_computers import IterativeGradientComputer  # 添加迭代式梯度计算
# 在文件开头添加以下导入
from trak.projectors import BasicProjector
from trak.gradient_computers import IterativeGradientComputer
from trak.modelout_functions import AbstractModelOutput
import tqdm


# 禁用CuDNN并设置确定性算法
torch.backends.cudnn.enabled = False
torch.use_deterministic_algorithms(True)

# 在文件开头添加以下环境变量配置
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 用于更清晰的错误定位


# 载入数据
def init_loaders(params):
    """
    初始化训练和验证数据加载器
    返回: (train_loader, valid_loaders)
    """
    # 创建数据集管理器
    dataset_params = {
        "use_ddp": params["training_params"]["use_ddp"],
        "batch_size": params["training_params"]["batch_size"],
        "num_gpu": params["training_params"]["nb_gpu"],
        "datasets": params["dataset_params"]["datasets"],

        "train": params["dataset_params"]["train"],
        "valid": params["dataset_params"]["valid"],
        "dataset_class": params["dataset_params"]["dataset_class"],
        "config": params["dataset_params"]["config"]
    }

    dataset_manager = DatasetManager(dataset_params)

    # 获取第一个验证集
    valid_name = next(iter(dataset_manager.valid_loaders.keys()))
    valid_loader = dataset_manager.valid_loaders[valid_name]

    # 返回训练和验证数据加载器
    return dataset_manager.train_loader, valid_loader


class FullOCRModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


if __name__ == "__main__":

    # 禁用CuDNN并设置确定性算法
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # 添加warn_only

    dataset_name = "IAM"  # ["RIMES", "IAM", "READ_2016"]

    params = {
        "dataset_params": {
            "datasets": {
                # dataset_name: "../../../Datasets/formatted/{}_lines".format(dataset_name),
                dataset_name: "/data/imucs_data/cw/ICCV2023/PR/major_revision_3/table_2_no_aug/cvl/data",
            },
            "train": {
                "name": "{}-train".format(dataset_name),
                "datasets": [dataset_name, ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [dataset_name, ],
            },
            "dataset_class": OCRDataset,
            "config": {
                "width_divisor": 8,  # Image width will be divided by 8
                "height_divisor": 32,  # Image height will be divided by 32
                "padding_value": 0,  # Image padding value
                "padding_token": 1000,  # Label padding value (None: default value is chosen)
                "charset_mode": "CTC",  # add blank token
                "constraints": ["CTC_line"],  # Padding for CTC requirements if necessary
                "preprocessings": [
                    {
                        "type": "dpi",  # modify image resolution
                        "source": 300,  # from 300 dpi
                        "target": 150,  # to 150 dpi
                    },
                    {
                        "type": "to_RGB",
                        # if grayscale image, produce RGB one (3 channels with same value) otherwise do nothing
                    },
                ],
                # Augmentation techniques to use at training time
                "augmentation": {
                    "dpi": {
                        "proba": 0,
                        "min_factor": 0.75,
                        "max_factor": 1.25,
                    },
                    "perspective": {
                        "proba": 0,
                        "min_factor": 0,
                        "max_factor": 0.3,
                    },
                    "elastic_distortion": {
                        "proba": 0,
                        "max_magnitude": 20,
                        "max_kernel": 3,
                    },
                    "random_transform": {
                        "proba": 0,
                        "max_val": 16,
                    },
                    "dilation_erosion": {
                        "proba": 0,
                        "min_kernel": 1,
                        "max_kernel": 3,
                        "iterations": 1,
                    },
                    "brightness": {
                        "proba": 0,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "contrast": {
                        "proba": 0,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "sign_flipping": {
                        "proba": 0,
                    },
                },
            }
        },

        "model_params": {
            # Model classes to use for each module
            "models": {
                "encoder": FCN_Encoder,
                "decoder": Decoder,
            },
            "transfer_learning": None,  # dict : {model_name: [state_dict_name, checkpoint_path, learnable, strict], }
            "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
            "dropout": 0.5,
            # dropout probability for standard dropout (half dropout probability is taken for spatial dropout)
        },

        "training_params": {
            "output_folder": "fcn_iam_line",  # folder names for logs and weigths
            "max_nb_epochs": 5000,  # max number of epochs for the training
            "max_training_time": 3600 * (10000),  # max training time limit (in seconds)
            "load_epoch": "best",  # ["best", "last"], to load weights from best epoch or last trained epoch
            "interval_save_weights": None,  # None: keep best and last only
            "use_ddp": False,  # Use DistributedDataParallel
            "use_apex": False,  # Enable mix-precision with apex package 原本是True
            # "nb_gpu": torch.cuda.device_count(),
            "nb_gpu": 4,
            "batch_size": 16,  # mini-batch size per GPU 原本是16
            "optimizer": {
                "class": Adam,
                "args": {
                    "lr": 0.0001,
                    "amsgrad": False,
                }
            },
            "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
            "eval_on_valid_interval": 10,  # Interval (in epochs) to evaluate during training
            "focus_metric": "cer",  # Metrics to focus on to determine best epoch
            "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
            "set_name_focus_metric": "{}-valid".format(dataset_name),
            "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
            "eval_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for evaluation on validation set during training
            "force_cpu": False,  # True for debug purposes to run on cpu only
        },
    }

    # 设置cuda在几卡上跑
    params["training_params"]["ddp_rank"] = 0

    # 初始化模型
    model = TrainerLineCTC(params)
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()
    device = 'cuda'
    full_model = FullOCRModel(model.models['encoder'], model.models['decoder']).to(device)

    # 初始化数据加载器
    train_loader, valid_loader = init_loaders(params)

    # 修改TRAKer初始化
    traker = TRAKer(
        model=full_model,
        task='ocr',
        train_set_size=len(train_loader.dataset),
        save_dir='./ocr_trak_results',
        device=device,  # 强制在CPU上执行
        proj_dim=1024,
        projector=BasicProjector,  # 显式使用基本投影器
        gradient_computer=IterativeGradientComputer,
        use_half_precision=False,  # 使用单精度提高稳定性
        lambda_reg=1e-5  # 添加正则化防止数值不稳定
    )

    # 在特征化循环前添加
    torch.set_grad_enabled(True)

    # 阶段1: 特征化（训练集）
    traker.load_checkpoint(full_model.state_dict(), model_id=0)
    # 修改特征化循环

    # 添加进度条
    total_batches = len(train_loader)
    progress_bar = tqdm.tqdm(total=total_batches, desc="traker.featurize", unit="batch")

    # # 添加计数器，只运行两个批次,到时候要删掉，现在为了调试！！
    # batch_count1 = 0
    # max_batches1 = 10  # 只运行两个批次

    for batch in train_loader:

        # if batch_count1 >= max_batches1:
        #     break  # 达到最大批次数量后跳出循环，为了调试！！

        images = batch['imgs'].to(device)
        labels = batch['labels'].to(device)
        label_lengths = torch.tensor(batch['labels_len'], dtype=torch.long).to(device)

        traker.featurize(
            batch=(images, labels, label_lengths),
            num_samples=images.size(0)
        )

        progress_bar.update(1)  # 更新进度条
        # batch_count1 += 1  # 更新批次计数器，为了调试！！

    progress_bar.close()  # 关闭进度条

    traker.finalize_features()


    # 阶段2: 评分（验证集）
    # 替换原有的评分阶段代码
    traker.start_scoring_checkpoint(
        exp_name='ocr_val',
        checkpoint=full_model.state_dict(),
        model_id=0,
        num_targets=len(valid_loader.dataset)
    )

    # 添加进度条
    total_batches = len(valid_loader)
    progress_bar = tqdm.tqdm(total=total_batches, desc="traker.score", unit="batch")

    # 添加计数器，只运行两个批次,到时候要删掉，现在为了调试！！
    # batch_count2 = 0
    # max_batches2 = 2  # 只运行两个批次

    for batch in valid_loader:

        # if batch_count2 >= max_batches2:
        #     break  # 达到最大批次数量后跳出循环，为了调试！！

        images = batch['imgs'].to(device)
        labels = batch['labels'].to(device)
        label_lengths = torch.tensor(batch['labels_len'], dtype=torch.long).to(device)

        # 正确解包传递三个参数
        traker.score(
            batch=(images, labels, label_lengths),
            num_samples=images.size(0)
        )

        progress_bar.update(1)  # 更新进度条
        # batch_count2 += 1  # 更新批次计数器，为了调试！！

    progress_bar.close()  # 关闭进度条

    # 获取最终影响分数
    scores = traker.finalize_scores(exp_name='ocr_val')

