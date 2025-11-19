import torch

from src.utils import load_yaml, set_seed
from src.path import SRC_PATH
from src.data import NGCFDataLoader
from model.ngcf import NGCF, trainer, evaluator  


# ---------------------------------------------------
# 실행 시작
# ---------------------------------------------------
if __name__ == "__main__":

    # ----------------------------
    # 1. Load config
    # ----------------------------
    config_fpath = SRC_PATH / "config.yaml"
    cfg = load_yaml(config_fpath)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    path_cfg = cfg["path"]

    # ----------------------------
    # 2. Seed
    # ----------------------------
    set_seed(data_cfg.get("seed", 42))

    # ----------------------------
    # 3. Device
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ----------------------------
    # 4. DataLoader
    # ----------------------------
    print("[INFO] Loading data...")
    dataloader = NGCFDataLoader(
        fname=data_cfg["fname"],
        source=data_cfg.get("source", "amazon"),
        test_size=data_cfg.get("test_size", 0.2),
        seed=data_cfg.get("seed", 42),
    )

    print(f"[INFO] #users: {dataloader.user_num}, #items: {dataloader.item_num}")
    print(f"[INFO] #train interactions: {len(dataloader.train_df)}")
    print(f"[INFO] #val   interactions: {len(dataloader.val_df)}")
    print(f"[INFO] #test  interactions: {len(dataloader.test_df)}")

    # sparse Laplacian
    L = dataloader.L.to(device)

    # ----------------------------
    # 5. Model 
    # ----------------------------
    model = NGCF(
        user_num=dataloader.user_num,
        item_num=dataloader.item_num,
        L=L,
        embed_dim=model_cfg["embed_dim"],
        n_layer=model_cfg["n_layer"],
        dropout=model_cfg["dropout"],
        l2_reg=model_cfg["l2_reg"],
        negative_slope=model_cfg.get("negative_slope", 0.2),
    ).to(device)

    print("[INFO] Model:")
    print(model)

    # ----------------------------
    # 6. Optimizer 
    # ----------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
    )

    # ----------------------------
    # 7. Train 
    # ----------------------------
    trainer(
        model=model,
        data_loader=dataloader,
        optimizer=optimizer,
        batch_size=train_cfg["batch_size"],
        epoch_num=train_cfg["epoch_num"],
        num_batches_per_epoch=train_cfg["num_batches_per_epoch"],
        eval_interval=train_cfg["eval_interval"],
        eval_k=eval_cfg["k"],
        patience=train_cfg["patience"],
        best_model_path=path_cfg["best_model_path"],
        device=device,
    )

    # ----------------------------
    # 8. Load Best model and test
    # ----------------------------
    print("[INFO] Loading best model and evaluating on TEST set...")
    best_state = torch.load(path_cfg["best_model_path"], map_location=device)
    model.load_state_dict(best_state)

    test_ndcg, test_hit = evaluator(
        model=model,
        data_loader=dataloader,
        k=eval_cfg["k"],
        device=device,
        is_test=True,  
    )

    print("========================================")
    print(f"[TEST] NDCG@{eval_cfg['k']}: {test_ndcg:.4f}, Hit@{eval_cfg['k']}: {test_hit:.4f}")
