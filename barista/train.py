import argparse
import copy

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch import nn, optim

from barista.data.braintreebank_dataset import BrainTreebankDataset
from barista.models.model import Barista
from barista.models.utils import seed_everything


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Barista model on BrainTreebank dataset"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="barista/config/braintreebank.yaml",
        help="Path to dataset configuration file",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="barista/config/train.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="barista/config/model.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="+",
        default=[],
        help="Override config parameters (e.g., --override epochs=50 optimization.finetune_lr=1e-4)",
    )
    return parser.parse_args()


def load_configs(args):
    """Load all configuration files."""
    dataset_config = OmegaConf.load(args.dataset_config)
    train_config = OmegaConf.load(args.train_config)
    model_config = OmegaConf.load(args.model_config)

    assert (
        len(dataset_config.finetune_sessions) == 1
    ), "Specify one session for finetuning"

    return dataset_config, train_config, model_config


def apply_overrides(config_dict, overrides):
    """Apply command-line overrides to configs using dot notation."""
    if not overrides:
        return config_dict

    override_dict = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: {override}. Expected format: key=value"
            )

        key, value = override.split("=", 1)

        try:
            if value.isnumeric():
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            elif value.startswith("[") or value in ("True", "False"):  # list, bool
                value = eval(value)
        except ValueError as e:
            print(e)
            pass

        keys = key.split(".")
        current = override_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    # Convert override dict to OmegaConf and merge
    override_conf = OmegaConf.create(override_dict)

    # Determine which config to merge based on keys
    merged_configs = {}
    for config_name, config in config_dict.items():
        config_keys = set(OmegaConf.to_container(config).keys())
        override_keys = set(override_dict.keys())

        if config_keys.intersection(override_keys):
            merged_configs[config_name] = OmegaConf.merge(config, override_conf)
        else:
            merged_configs[config_name] = config

    if merged_configs.get("train") is not None:
        merged_configs["train"] = OmegaConf.merge(
            merged_configs["train"], override_conf
        )

    return merged_configs


def setup_dataloaders(dataset_config, train_config):
    """Initialize dataset and create dataloaders."""
    dataset = BrainTreebankDataset(dataset_config)

    train_dataloader = dataset.get_dataloader("train", train_config)
    val_dataloader = dataset.get_dataloader("val", train_config)
    test_dataloader = dataset.get_dataloader("test", train_config)

    print(f"Train: {len(train_dataloader.dataset.metadata)} samples")
    print(f"Val: {len(val_dataloader.dataset.metadata)} samples")
    print(f"Test: {len(test_dataloader.dataset.metadata)} samples")

    dataset.check_no_common_segment(train_dataloader, val_dataloader, test_dataloader)

    return dataset, train_dataloader, val_dataloader, test_dataloader


def get_optimizer(model, finetune_lr=1e-4, new_param_lr=1e-3):
    """Create optimizer with different learning rates for task and upstream parameters."""
    task_params, upstream_params = [], []

    for _, p in model.get_task_params():
        if p.requires_grad:
            task_params.append(p)

    for _, p in model.get_upstream_params():
        if p.requires_grad:
            upstream_params.append(p)

    params = [
        {"params": upstream_params, "lr": finetune_lr},
        {"params": task_params, "lr": new_param_lr},
    ]

    optimizer = optim.AdamW(params, lr=finetune_lr, weight_decay=1e-2)
    return optimizer


def get_lr_scheduler(optimizer):
    """Create learning rate scheduler with warmup and exponential decay."""
    milestone = 5

    lr_schedulers_list = [
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=milestone,
        ),
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99),
    ]

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        lr_schedulers_list,
        milestones=[milestone],
    )
    return lr_scheduler


def load_pretrained_weights(model, checkpoint_path, device):
    """Load pretrained weights, excluding masked_recon and multi_head_fc layers."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    print(f"Pretrained weights loaded from {checkpoint_path}")
    return model


def freeze_tokenizer(model):
    for n, p in model.tokenizer.named_parameters():
        p.requires_grad = False


def print_number_of_parmas(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model parameters: {total_params}\t Trainable params: {trainable_params}")


def run_epoch(
    model, dataloader, criterion, device, optimizer=None, scheduler=None, train=False
):
    """Run one epoch of training or evaluation."""
    if train:
        model.train()
    else:
        model.eval()

    all_preds = []
    all_labels = []
    running_loss = 0

    for batch in dataloader:
        x = [x_item.to(device) for x_item in batch.x]
        y = batch.labels.flatten().long().to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(
                x,
                subject_sessions=batch.subject_sessions,
            )
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * y.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labels = y.detach().cpu().numpy()

        all_preds.append(probs)
        all_labels.append(labels)

    if train:
        # step scheduler at epoch interval
        scheduler.step()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = float("nan")

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss, auc


def finetune_model(model, train_dataloader, val_dataloader, train_config, device):
    """Finetune the model and track best validation performance."""
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        model,
        finetune_lr=train_config.optimization.finetune_lr,
        new_param_lr=train_config.optimization.new_param_lr,
    )
    scheduler = get_lr_scheduler(optimizer)

    best_val_auc = -1
    best_state = None
    num_epochs = train_config.epochs

    for epoch in range(num_epochs):
        train_loss, train_auc = run_epoch(
            model, train_dataloader, criterion, device, optimizer, scheduler, train=True
        )
        val_loss, val_auc = evaluate_model(model, val_dataloader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} "
            f"- Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}"
        )

        # Track best model by validation AUC
        if best_state is None or val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "epoch": epoch + 1,
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "scheduler": copy.deepcopy(scheduler.state_dict()),
                "val_auc": val_auc,
            }

    return best_state, criterion


def evaluate_model(model, test_dataloader, criterion, device):
    """Evaluate model on test set."""
    test_loss, test_auc = run_epoch(
        model, test_dataloader, criterion, device, train=False
    )
    return test_loss, test_auc


def main():
    """Main training pipeline."""
    # Parse arguments and load configs
    args = parse_args()
    dataset_config, train_config, model_config = load_configs(args)

    configs = {"dataset": dataset_config, "train": train_config, "model": model_config}
    configs = apply_overrides(configs, args.override)
    dataset_config = configs["dataset"]
    train_config = configs["train"]
    model_config = configs["model"]

    # Set random seed
    seed_everything(train_config.seed)

    # Setup data
    dataset, train_dataloader, val_dataloader, test_dataloader = setup_dataloaders(
        dataset_config, train_config
    )

    # Get fine-tuning session info
    ft_session = dataset_config.finetune_sessions[0]
    ft_session_n_chans = dataset.metadata.get_subject_session_full_d_data()[ft_session][
        -1
    ]

    # Initialize model
    device = train_config.device
    model = Barista(model_config, dataset.metadata)

    # Load pretrained weights
    if train_config.checkpoint_path:
        print("Running pretrained model")
        model = load_pretrained_weights(model, train_config.checkpoint_path, device)

        # Freeze tokenizer
        if train_config.optimization.freeze_tokenizer:
            freeze_tokenizer(model)

    else:
        print("Running non-pretrained model")

    # Create downstream head and move to device
    model.create_downstream_head(n_chans=ft_session_n_chans, output_dim=2)
    model.to(device)

    print_number_of_parmas(model)

    # Finetune model
    best_state, criterion = finetune_model(
        model, train_dataloader, val_dataloader, train_config, device
    )
    print(f"\nBEST VAL AUC: {best_state['val_auc']:.4f}")

    # Evaluate on test set
    _, last_test_auc = evaluate_model(model, test_dataloader, criterion, device)
    print(f"LAST TEST AUC: {last_test_auc:.4f}")

    # Load best model for testing
    model.load_state_dict(best_state["model"])

    # Evaluate on test set
    _, test_auc = evaluate_model(model, test_dataloader, criterion, device)

    print(f"BEST TEST AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
