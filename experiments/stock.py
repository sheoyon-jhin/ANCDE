import common
import datasets
import os
import numpy as np
import random
import torch
import pandas as pd
import torch.utils.data as data_utils
from random import SystemRandom

from parse import parse_args

from time_dataset import TimeDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

args = parse_args()


def main(
    dataset_name=args.dataset_name,
    manual_seed=args.seed,
    missing_rate=args.missing_rate,  # dataset parameters
    device="cuda",
    max_epochs=300,
    *,  # training parameters
    model_name=args.model,
    hidden_channels=args.h_channels,
    hidden_hidden_channels=args.hh_channels,
    num_hidden_layers=args.layers,
    lr=args.lr,
    slope_check=args.slope_check,
    soft=args.soft,
    timewise=args.timewise,
    attention_channel=args.attention_channel,
    attention_attention_channel=args.attention_attention_channel,
    sequence=args.sequence,
    step_mode=args.step_mode,
    dry_run=False,
    **kwargs,
):

    batch_size = 200
    lr = lr * (batch_size / 32)
    PATH = os.path.dirname(os.path.abspath(__file__))
    data_path = PATH + "/datasets/stock_data.csv"

    # input_channels = data[0].shape[1] - 1
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    # if you are using GPU
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    intensity_data = True if model_name in ("odernn", "dt", "decay") else False
    (
        times,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        num_classes,
        input_channels,
    ) = datasets.stock.get_data(
        data_path,
        sequence,
        missing_rate,
        device,
        intensity=intensity_data,
        batch_size=batch_size,
    )

    output_channels = 1
    num_classes = 1
    # if num_classes == 2:
    #     output_channels = 1
    # else:
    #     output_channels = num_classes
    print(input_channels)
    experiment_id = int(SystemRandom().random() * 100000)
    file = PATH + "/" + "Google_h_prime/" + f"{experiment_id}.npy"
    make_model = common.make_model(
        model_name,
        input_channels,
        output_channels,
        hidden_channels,
        hidden_hidden_channels,
        attention_channel,
        attention_attention_channel,
        num_hidden_layers,
        use_intensity=False,
        slope_check=slope_check,
        soft=soft,
        timewise=timewise,
        rtol=args.rtol,
        atol=args.atol,
        file=file,
        initial=True,
    )

    if dry_run:
        name = None
    else:
        name = dataset_name + str(int(missing_rate * 100))

    experiments = "stock0" + str(manual_seed)
    return common.main_forecasting(
        experiments,
        model_name,
        name,
        times,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
        make_model,
        num_classes,
        max_epochs,
        lr,
        slope_check,
        kwargs,
        step_mode=step_mode,
    )


def run_all(
    group, device, dataset_name, model_names=("ncde", "odernn", "dt", "decay", "gruode")
):
    if group == 1:
        missing_rate = 0.3
    elif group == 2:
        missing_rate = 0.5
    elif group == 3:
        missing_rate = 0.7
    else:
        raise ValueError
    model_kwargs = dict(
        ncde=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
        odernn=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
        dt=dict(
            hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None
        ),
        decay=dict(
            hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None
        ),
        gruode=dict(
            hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None
        ),
    )
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(
                dataset_name,
                missing_rate,
                device,
                model_name=model_name,
                **model_kwargs[model_name],
            )


if __name__ == "__main__":
    # main(dataset_name ="CharacterTrajectories")
    # main(dataset_name="EigenWorms")

    main()
