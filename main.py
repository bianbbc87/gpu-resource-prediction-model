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
    # Logger (EARLY)
    # -------------------------------------------------
    logger = build_logger("PerfSeer")
    logger.info("ENTER main()")
    logger.info(f"Args: config={args.config}, output_dir={args.output_dir}")
    
    # -------------------------------------------------
    # Load config (local or gs://)
    # -------------------------------------------------
    logger.info("Before load_config()")
    cfg = load_config(args.config)
    logger.info("After load_config()")

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["dataset"]
    device_cfg = cfg["device"]

    logger.info(f"Train config: {train_cfg}")
    logger.info(f"Dataset config: {data_cfg}")
    logger.info(f"Model config: {model_cfg}")

    # -------------------------------------------------
    # Device (infra decides)
    # -------------------------------------------------
    device = device_cfg.get("device", "cuda")
    logger.info(f"Using device: {device}")

    # -------------------------------------------------
    # Output directory (MUST be local path)
    # -------------------------------------------------
    output_dir = args.output_dir
    logger.info(f"Before os.makedirs({output_dir})")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("After os.makedirs()")

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # -------------------------------------------------
    # Dataset / Dataloader
    # -------------------------------------------------
    logger.info("Before Dataset init")

    dataset = PerfGraphDataset(
        graph_dir=data_cfg["graph_dir"],
        label_dir=data_cfg["label_dir"],
        max_samples=data_cfg.get("max_samples", None),
    )

    logger.info("After Dataset init")

    dataloader = build_dataloader(
        dataset,
        batch_size=train_cfg.get("batch_size", 1),
        shuffle=True,
    )

    # -------------------------------------------------
    # Model (auto-detect dimensions from data)
    # -------------------------------------------------
    logger.info("===== Start Modeling =====")
    
    # Get first sample to detect dimensions
    sample_perfgraph, _ = dataset[0]
    node_dim = sample_perfgraph.V.shape[1]
    edge_dim = sample_perfgraph.E.shape[1] 
    global_dim = sample_perfgraph.u.shape[0]
    
    logger.info(f"Auto-detected dimensions: node_dim={node_dim}, edge_dim={edge_dim}, global_dim={global_dim}")
    
    model = SeerNet(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        global_node_dim=model_cfg.get("global_node_dim", 256),
        output_dim=1,  # execution time
    )
    
    logger.info("===== Model Created =====")
    
    model.to(device)
    
    logger.info("===== Model Moved to Device =====")

    # -------------------------------------------------
    # Optimizer / Loss
    # -------------------------------------------------
    logger.info("===== Setup Optimizer / Loss =====")
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
    )
    
    logger.info("===== Optimizer Created =====")
    
    loss_fn = build_loss(train_cfg.get("loss", "mse"))
    
    logger.info("===== Loss Function Created =====")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    
    logger.info("===== Trainer Ready =====")

    evaluator = Evaluator(
        model=model,
        device=device,
    )
    
    logger.info("===== Evaluator Ready =====")

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    logger.info("===== Start Training =====")

    epochs = train_cfg["epochs"]

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        logger.info(f"--- Epoch {epoch:03d} ---")

        for batch_idx, (perfgraph, label) in enumerate(dataloader, 1):
            loss = trainer.train_step(perfgraph, label)
            epoch_loss += loss
            
            # 배치별 로그 (100배치마다)
            if batch_idx % 100 == 0 or batch_idx == len(dataloader):
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.6f}")

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
    
    logger.info("===== Saving Results =====")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "model.pt"),
    )

    logger.info("Results saved successfully")


if __name__ == "__main__":
    main()
