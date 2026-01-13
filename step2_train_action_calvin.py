import torch.nn as nn
# import cv2
from torchvision.utils import save_image

import logging
from pathlib import Path
import sys
from typing import List, Union
import os
import wandb
from time import time
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch
from glob import glob
from copy import deepcopy
from collections import OrderedDict

from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer

from policy_models.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info,
)
#from torch.nn.parallel import DistributedDataParallel as DDP

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


#@hydra.main(config_path="./policy_conf", config_name="VPP_Calvinabc_train")
def train(cfg: DictConfig, wandb_opts: dict | None = None) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    # Enable DDP to tolerate legitimately unused parameters (e.g., optional heads)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = accelerator.device
    # new added
    torch.set_float32_matmul_precision('medium')


    if accelerator.is_main_process:
        os.makedirs(cfg.log_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        from datetime import datetime
        uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #experiment_dir = f"{cfg.log_dir}/{uuid}"  # Create an experiment folder
        experiment_dir = "cvpr2025"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        eval_dir = f"{experiment_dir}/eval"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
        # Initialize Weights & Biases if requested
        try:
            wb = wandb_opts or {}
            use_wandb = bool(wb.get("use", False))
            if use_wandb:
                cfg_d = OmegaConf.to_container(cfg, resolve=True)
                project = wb.get("project") or ((cfg_d.get("logger", {}) or {}).get("project") or cfg_d.get("benchmark_name") or "vpp")
                group = wb.get("group") or ((cfg_d.get("logger", {}) or {}).get("group") or "models")
                name = wb.get("name") or f"calvin_step2_{uuid}"
                mode = wb.get("mode") or "online"
                if mode == "offline":
                    os.environ["WANDB_MODE"] = "offline"
                elif mode == "disabled":
                    os.environ["WANDB_DISABLED"] = "true"
                wandb.init(project=project, group=group, name=name,
                           config=cfg_d, dir=experiment_dir, reinit=True)
                logger.info(f"Initialized Weights & Biases: project={project}, group={group}, name={name}, mode={mode}")
        except Exception as ex:
            logger.info(f"wandb init failed or disabled: {ex}")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    if accelerator.is_main_process:
        logger.info(f"Global batch size {cfg.batch_size:,} num_processes ({accelerator.num_processes})")
    chk = get_last_checkpoint(Path.cwd())
    train_loader = datamodule.train_dataloader()["lang"]
    test_loader = datamodule.val_dataloader()["lang"]
    # chk = get_last_checkpoint(Path('/home/temp_store/code/calvin_d/logs/runs/2023-09-10/17-52-50/saved_models/epoch=09_eval_lh/avg_seq_len=2.62.ckpt'))
    # Load Model
    model = hydra.utils.instantiate(cfg.model)
    if "pretrain_chk" in cfg:
        initialize_pretrained_weights(model, cfg)

    if cfg.use_ckpt_path:
        state_dict = torch.load(cfg.ckpt_path, map_location='cpu')
        # print('state_dict_key:', state_dict['model'].keys())
        print('load_from_ckpt:',cfg.ckpt_path)
        # c = []
        # hydra.initialize(config_path="../../conf")
        # hydra.main(config_name="config_abc.yaml")(lambda x: c.append(x))()
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict['model'])

    model = model.to(device)
    model.process_device()


    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = model.configure_optimizers()["optimizer"]
    Ir_scheduler = model.configure_optimizers()["lr_scheduler"]["scheduler"]

    model.on_train_start()
    if accelerator.is_main_process:
        logger.info(f"model parameter init")
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()
    model.train()
    model, opt, loader = accelerator.prepare(model, opt, train_loader)
   # model = DDP(model, find_unused_parameters=True)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_action = 0.0
    running_contra = 0.0
    running_proto = 0.0
    running_metric = 0.0
    start_time = time()
    eval_batch = None
    best_eval_loss = 1e8

    if accelerator.is_main_process:
        logger.info(f"Training for {cfg.max_epochs} epochs...")

    for epoch in range(cfg.max_epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        running_loss = 0

        for idx,data_batch in enumerate(loader):
            with accelerator.autocast():
                loss = model(data_batch)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            Ir_scheduler.step()
            update_ema(ema, model)
            running_loss += loss
            # collect component losses for logging
            try:
                _m = model.module if accelerator.num_processes > 1 else model
                running_action += float(getattr(_m, "_last_action_loss", 0.0))
                running_contra += float(getattr(_m, "_last_contra_loss", 0.0))
                running_proto += float(getattr(_m, "_last_proto_loss", 0.0))
                running_metric += float(getattr(_m, "_last_metric_loss", 0.0))
            except Exception:
                pass
            log_steps += 1
            train_steps += 1
            if train_steps % cfg.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = (running_loss / log_steps)
                avg_loss = avg_loss.detach().item() if torch.is_tensor(avg_loss) else float(avg_loss)
                # component avgs
                avg_action = running_action / max(1, log_steps)
                avg_contra = running_contra / max(1, log_steps)
                avg_proto = running_proto / max(1, log_steps)
                avg_metric = running_metric / max(1, log_steps)

                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss total={avg_loss:.6f} | action={avg_action:.6f} contra={avg_contra:.6f} proto={avg_proto:.6f} metric={avg_metric:.6f} | {steps_per_sec:.2f} steps/s")
                    # wandb logging (main process only)
                    if (wandb_opts or {}).get("use", False) and wandb.run is not None:
                        wandb.log({
                            "train/total_loss": avg_loss,
                            "train/action_loss": avg_action,
                            "train/contrastive_loss": avg_contra,
                            "train/proto_loss": avg_proto,
                            "train/metric_loss": avg_metric,
                            "train/steps_per_sec": steps_per_sec,
                            "epoch": epoch,
                            "step": train_steps,
                        }, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                running_action = 0.0
                running_contra = 0.0
                running_proto = 0.0
                running_metric = 0.0
                log_steps = 0
                start_time = time()
        if accelerator.is_main_process:
            model.eval()
            logger.info(f"Finished training epoch {epoch}")
            logger.info(f"started validation epoch {epoch}")
            total_val_loss = 0
            for test_batch in test_loader:
                val_loss=model.module.validation_step(test_batch)
                total_val_loss += val_loss["validation_loss"]
                log_steps += 1
            model.train()
        # Run KMeans update at epoch end across all ranks to gather distributed features
        if cfg.model.use_kmeans and ((epoch + 1) % cfg.model.kmeans_refresh_interval == 0):
            try:
                _m = model.module if accelerator.num_processes > 1 else model
                # use sharded training loader to ensure each rank processes its shard
                if accelerator.is_main_process:
                    logger.info("Running KMeans full update over training loader shard...")
                _m.run_kmeans_with_loader(loader)
            except Exception as ex:
                if accelerator.is_main_process:
                    logger.info(f"KMeans update skipped due to error: {ex}")
        if accelerator.is_main_process:
            total_val_loss = total_val_loss/log_steps
            log_steps = 0
            # wandb validation logging
            if (wandb_opts or {}).get("use", False) and wandb.run is not None:
                wandb.log({
                    "val/total_loss": float(total_val_loss),
                    "epoch": epoch,
                    "step": train_steps,
                }, step=train_steps)
            checkpoint = {
                "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                # "ema": ema.state_dict(),
                # "opt": opt.state_dict(),
                "args": cfg,
            }
            if total_val_loss < best_eval_loss:
                # if not args.without_ema:
                #     checkpoint["ema"] = ema.state_dict()
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}_{total_val_loss:.3f}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                best_eval_loss = total_val_loss
                if (wandb_opts or {}).get("use", False) and wandb.run is not None:
                    wandb.summary["best_val_loss"] = float(best_eval_loss)
                    wandb.summary["best_ckpt_path"] = checkpoint_path
            last_path = f"{checkpoint_dir}/last.pt"
            torch.save(checkpoint, last_path)


    # Setup accelerator:

def setup_logger(cfg: DictConfig, model: LightningModule):
    """
    Set up the logger (tensorboard or wandb) from hydra config.

    Args:
        cfg: Hydra config
        model: LightningModule

    Returns:
        logger
    """
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = pathlib_cwd.parent.name + "/" + pathlib_cwd.name
        cfg.logger.id = cfg.logger.name.replace("/", "_")
        train_logger = hydra.utils.instantiate(cfg.logger)
        # train_logger.watch(model)
    else:
        train_logger = hydra.utils.instantiate(cfg.logger)
    return train_logger

if __name__ == "__main__":
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--text_encoder_path", type=str, default="")
    parser.add_argument("--root_data_dir", type=str, default="")
    parser.add_argument("--lang_folder", type=str, default="lang_annotations",
                        help="Folder name under each scene containing auto_lang_ann.npy (e.g., 'lang_annotations' for debug dataset)")
    parser.add_argument("--task_index_json", type=str, default="",
                        help="Path to JSON built by scripts/build_calvin_task_index.py for skill/task ids. If set, skill_id will be injected into batches.")
    # KMeans options
    parser.add_argument("--use_kmeans", action="store_true", help="Enable K-Means clustering for prototype losses")
    parser.add_argument("--kmeans_k", type=int, default=50, help="Number of K-Means clusters")
    parser.add_argument("--kmeans_refresh_interval", type=int, default=1, help="Epoch interval to refresh K-Means over full dataset")
    parser.add_argument("--kmeans_loader_key", type=str, default="lang", help="Which train loader key to use for K-Means feature extraction")
    # Loss weight options
    parser.add_argument("--lambda_contra", type=float, default=None, help="Weight for contrastive loss")
    parser.add_argument("--lambda_proto", type=float, default=None, help="Weight for prototype loss")
    parser.add_argument("--lambda_metric", type=float, default=None, help="Weight for metric loss")
    # Weights & Biases options
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="", help="Wandb project name override")
    parser.add_argument("--wandb_group", type=str, default="", help="Wandb group override")
    parser.add_argument("--wandb_name", type=str, default="", help="Wandb run name override")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="Wandb mode")

    args = parser.parse_args()
    from hydra import compose, initialize

    with initialize(config_path="./policy_conf", job_name="VPP_Calvinabc_train"):
        cfg = compose(config_name="VPP_Calvinabc_train")
    # Validate task index json path if provided
    if args.task_index_json and not os.path.isfile(args.task_index_json):
        print(f"[warn] task_index_json not found: {args.task_index_json}. Disabling skill-id injection.")
        args.task_index_json = ""
    if args.video_model_path:
        cfg.model.pretrained_model_path = args.video_model_path
    if args.text_encoder_path:
        cfg.model.text_encoder_path = args.text_encoder_path
    cfg.root_data_dir = args.root_data_dir
    cfg.datamodule.root_data_dir = args.root_data_dir
    # Allow overriding language folder for debug dataset compatibility
    cfg.lang_folder = args.lang_folder
    # Pass task index json to datamodule if provided
    if hasattr(cfg, 'datamodule'):
        if args.task_index_json:
            cfg.datamodule.task_index_json = args.task_index_json
            cfg.datamodule.use_skill_id = True
        else:
            # ensure defaults if not present in config
            if not hasattr(cfg.datamodule, 'use_skill_id'):
                cfg.datamodule.use_skill_id = False
            if not hasattr(cfg.datamodule, 'task_index_json'):
                cfg.datamodule.task_index_json = ""
    # KMeans config from CLI
    cfg.model.use_kmeans = bool(args.use_kmeans)
    cfg.model.kmeans_k = int(args.kmeans_k)
    cfg.model.kmeans_refresh_interval = int(args.kmeans_refresh_interval)
    cfg.model.kmeans_loader_key = str(args.kmeans_loader_key)
    # Loss weights from CLI (keep defaults if None)
    if args.lambda_contra is not None:
        cfg.model.lambda_contra = float(args.lambda_contra)
    if args.lambda_proto is not None:
        cfg.model.lambda_proto = float(args.lambda_proto)
    if args.lambda_metric is not None:
        cfg.model.lambda_metric = float(args.lambda_metric)
    # Build wandb options without touching the structured cfg
    wandb_opts = {
        "use": bool(args.use_wandb),
        "project": args.wandb_project or None,
        "group": args.wandb_group or None,
        "name": args.wandb_name or None,
        "mode": args.wandb_mode or "online",
    }
    train(cfg, wandb_opts)
