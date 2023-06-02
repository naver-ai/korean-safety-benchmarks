import argparse
from src.model import ClassificationModel

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", type=str)
    parser.add_argument("--seed", type=int, default=42)

    ## WANDB
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)

    ## GPU
    parser.add_argument(
        "--gpu_accelerator", type=str, default="gpu", help="Set 'cpu' for debugging"
    )
    parser.add_argument(
        "--gpu_strategy", type=str, default="ddp", help="Options: null, ddp, dp, ..."
    )
    parser.add_argument("--gpu_devices", type=int, default=1)

    ## CHECKPOINT
    parser.add_argument("--checkpoint_dirpath", type=str, default="data/models")
    parser.add_argument(
        "--checkpoint_filename", type=str, default="ckpt-{epoch:03d}-{val_loss:.5f}"
    )
    parser.add_argument("--checkpoint_save_top_k", type=int, default=1)
    parser.add_argument("--checkpoint_monitor", type=str, default="val_loss")

    ## MODE
    parser.add_argument("--do_train_only", type=str2bool, default=False)
    parser.add_argument("--do_test_only", type=str2bool, default=False)    

    ## VALIDATION
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--earlystop_patience", type=int, default=5)

    ## LOG
    parser.add_argument("--log_interval", type=int, default=20)

    parser = ClassificationModel.add_model_specific_args(parser)

    args = parser.parse_args()
    return args
