# 실행 진입점

## 1. ONNX 로드

## 2. PerfGraph 생성

## 3. SeerNet forward

## 4. train / eval 분기

import argparse
import os
import json
import torch
import torch.optim as optim

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import build_logger

from train import (
    PerfGraphDataset,
    build_dataloader,
    build_loss,
    Trainer,
    Evaluator,
)

from model.seernet import SeerNet
# from model.seernet import SeerNetMulti  # multi-metric 실험 시 사용


def parse_args():
    parser = argparse.ArgumentParser("PerfSeer Research Entry Point")

    # paths
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # experiment control
    parser.add_argument("--seed", type=int, default=42)

    # overrides (Argo parameters)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)

    # data path override (PVC 대응)
    parser.add_argument("--graph_dir", type=str)
    parser.add_argument("--label_dir", type=str)

    return parser.parse_args()


# -------------------------------------------------
# Argument parsing
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("PerfSeer Research Entry Point")

    # required paths
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # experiment control
    parser.add_argument("--seed", type=int, default=42)

    # runtime overrides (Argo-friendly)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)

    # dataset path override (PVC 대응)
    parser.add_argument("--graph_dir", type=str)
    parser.add_argument("--label_dir", type=str)

    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()

    # -------------------------------------------------
    # Load config
    # -------------------------------------------------
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["dataset"]

    # -------------------------------------------------
    # Override config by args (runtime control)
    # -------------------------------------------------
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    if args.graph_dir is not None:
        data_cfg["graph_dir"] = args.graph_dir
    if args.label_dir is not None:
        data_cfg["label_dir"] = args.label_dir

    # -------------------------------------------------
    # Device (infra decides, not config)
    # -------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------
    # Setup
    # -------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    logger = build_logger("PerfSeer")
    logger.info(f"Using device: {device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Train config: {train_cfg}")
    logger.info(f"Dataset config: {data_cfg}")

    set_seed(args.seed)

    # -------------------------------------------------
    # Dataset / Dataloader
    # -------------------------------------------------
    dataset = PerfGraphDataset(
        graph_dir=data_cfg["graph_dir"],
        label_dir=data_cfg["label_dir"],
    )

    dataloader = build_dataloader(
        dataset,
        batch_size=train_cfg.get("batch_size", 1),  # graph-level default
        shuffle=True,
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = SeerNet(
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        global_dim=model_cfg["global_dim"],
        hidden_dim=model_cfg.get("hidden_dim", 256),
        global_node_dim=model_cfg.get("global_node_dim", 256),
        output_dim=1,  # execution time
    )

    model.to(device)

    # -------------------------------------------------
    # Optimizer / Loss
    # -------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
    )

    loss_fn = build_loss(train_cfg.get("loss", "mse"))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    evaluator = Evaluator(
        model=model,
        device=device,
    )

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    logger.info("===== Start Training =====")

    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_loss = 0.0

        for perfgraph, label in dataloader:
            loss = trainer.train_step(perfgraph, label)
            epoch_loss += loss

        avg_loss = epoch_loss / len(dataloader)
        metrics = evaluator.evaluate(dataloader)

        logger.info(
            f"[Epoch {epoch:03d}] "
            f"Loss={avg_loss:.6f} | "
            f"MAPE={metrics['MAPE']:.4f} | "
            f"RMSPE={metrics['RMSPE']:.4f}"
        )

    logger.info("===== Training Finished =====")

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    result = {
        "seed": args.seed,
        "final_mape": metrics["MAPE"],
        "final_rmspe": metrics["RMSPE"],
        "config": cfg,
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    torch.save(
        model.state_dict(),
        os.path.join(args.output_dir, "model.pt"),
    )

    logger.info("Results saved successfully")


if __name__ == "__main__":
    main()