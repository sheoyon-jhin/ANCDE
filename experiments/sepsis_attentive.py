import common
import datasets
import os
import numpy as np
import random
import torch
from random import SystemRandom
from parse import parse_args

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

args = parse_args()


class InitialValueNetwork(torch.nn.Module):
    def __init__(self, intensity, hidden_channels, model, slope_check):
        super(InitialValueNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(7 if intensity else 5, 256)
        self.linear2 = torch.nn.Linear(256, hidden_channels)

        self.model = model
        self.slope_check = slope_check

    def forward(self, times, coeffs, final_index, slope_check, **kwargs):
        # import pdb ; pdb.set_trace()
        *coeffs, static = coeffs
        z0 = self.linear1(static)
        z0 = z0.relu()
        z0 = self.linear2(z0)
        return self.model(times, coeffs, final_index, slope_check, z0=z0, **kwargs)


def main(
    manual_seed=args.seed,
    intensity=args.intensity,  # Whether to include intensity or not
    device="cuda",
    max_epochs=300,
    pos_weight=10,
    *,
    model_name=args.model,
    hidden_channels=args.h_channels,
    hidden_hidden_channels=args.hh_channels,
    num_hidden_layers=args.layers,
    lr=args.lr,
    slope_check=args.slope_check,
    soft=args.soft,
    timewise=args.timewise,  # model parameters
    attention_channel=args.attention_channel,
    attention_attention_channel=args.attention_attention_channel,
    step_mode=args.step_mode,
    dry_run=False,
    c1=args.c1,
    c2=args.c2,
    **kwargs,
):

    batch_size = 1024
    lr = lr * (batch_size / 32)
    PATH = os.path.dirname(os.path.abspath(__file__))

    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    static_intensity = intensity

    time_intensity = intensity or (model_name in ("odernn", "dt", "decay"))

    times, train_dataloader, val_dataloader, test_dataloader = datasets.sepsis.get_data(
        static_intensity, time_intensity, batch_size
    )

    input_channels = 1 + (1 + time_intensity) * 34
    experiment_id = int(SystemRandom().random() * 100000)
    file = PATH + "/" + "Sepsis_h_prime/" + f"{experiment_id}.npy"
    SAVED_PATH = PATH + "/" + "Sepsis_h_prime/"
    if not os.path.exists(SAVED_PATH):
        os.makedirs(SAVED_PATH)
    make_model = common.make_model(
        model_name,
        input_channels,
        1,
        hidden_channels,
        hidden_hidden_channels,
        attention_channel,
        attention_attention_channel,
        num_hidden_layers,
        use_intensity=intensity,
        slope_check=slope_check,
        soft=soft,
        timewise=timewise,
        rtol=args.rtol,
        atol=args.atol,
        file=file,
        initial=False,
    )

    def new_make_model():
        model, regularise, regularise2 = make_model()
        model.linear.weight.register_hook(lambda grad: 100 * grad)
        model.linear.bias.register_hook(lambda grad: 100 * grad)
        return (
            InitialValueNetwork(intensity, input_channels, model, slope_check),
            regularise,
            regularise2,
        )

    if dry_run:
        name = None
    else:
        intensity_str = "_intensity" if intensity else "_nointensity"
        name = "sepsis" + intensity_str
    num_classes = 2
    experiments = "sepsis" + str(manual_seed)

    return common.main(
        experiments,
        model_name,
        name,
        times,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
        new_make_model,
        num_classes,
        max_epochs,
        lr,
        slope_check,
        kwargs,
        pos_weight=torch.tensor(pos_weight),
        step_mode=step_mode,
        c1=args.c1,
        c2=args.c2,
    )


def run_all(intensity, device, model_names=("ncde", "odernn", "dt", "decay", "gruode")):
    model_kwargs = dict(
        ncde=dict(hidden_channels=49, hidden_hidden_channels=49, num_hidden_layers=4),
        odernn=dict(
            hidden_channels=128, hidden_hidden_channels=128, num_hidden_layers=4
        ),
        dt=dict(
            hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None
        ),
        decay=dict(
            hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None
        ),
        gruode=dict(
            hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None
        ),
    )
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(intensity, device, model_name=model_name, **model_kwargs[model_name])


if __name__ == "__main__":
    main()
