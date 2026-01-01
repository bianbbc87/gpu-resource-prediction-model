# 실행 진입점

## 1. ONNX 로드

## 2. PerfGraph 생성

## 3. SeerNet forward

## 4. train / eval 분기

# =====================================
# PerfSeer Research Entry Point
# =====================================

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


# -------------------------------------------------
# Argument parsing (minimal, stable)
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("PerfSeer Research Entry Point")

    # required
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()

    # -------------------------------------------------
    # Load config (local or gs://)
    # -------------------------------------------------
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["dataset"]

    # -------------------------------------------------
    # Device (infra decides)
    # -------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------
    # Output directory (MUST be local path)
    # -------------------------------------------------
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------
    # Logger / Seed
    # -------------------------------------------------
    logger = build_logger("PerfSeer")
    logger.info(f"Using device: {device}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Train config: {train_cfg}")
    logger.info(f"Dataset config: {data_cfg}")

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # -------------------------------------------------
    # Dataset / Dataloader
    # -------------------------------------------------
    dataset = PerfGraphDataset(
        graph_dir=data_cfg["graph_dir"],
        label_dir=data_cfg["label_dir"],
    )

    dataloader = build_dataloader(
        dataset,
        batch_size=train_cfg.get("batch_size", 1),
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

    epochs = train_cfg["epochs"]

    for epoch in range(1, epochs + 1):
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
    # Save results (LOCAL ONLY)
    # -------------------------------------------------
    result = {
        "seed": seed,
        "final_mape": metrics["MAPE"],
        "final_rmspe": metrics["RMSPE"],
        "config": cfg,
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "model.pt"),
    )

    logger.info("Results saved successfully")


if __name__ == "__main__":
    main()
