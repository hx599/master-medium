import argparse
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from hzscm_common import (
    DEFAULT_METHOD_PATCH,
    THIS_DIR,
    MultiModalPatchDataset,
    aggregate_results,
    build_known_train_full_test_split,
    compute_metrics,
    compute_seen_unseen_metrics,
    load_dataset,
    project_cube_pca,
    save_csv,
    set_seed,
    temp_sys_path,
)
from mft_model_hzscm import MFT


def progress(iterable, *, total=None, desc="", leave=True):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Run comparison baselines with HZSCM hidden-class training / full-class testing.")
    parser.add_argument("--method", required=True, choices=["mft", "ncglf2", "exvit", "frit", "s2crossmamba", "s2mamba", "udm", "unet"])
    parser.add_argument("--dataset", default="MUUFL", choices=["Houston", "HS-SAR-Berlin", "MUUFL"])
    parser.add_argument("--data_root", default="D:/Master_medium/dataset")
    parser.add_argument("--del_class", nargs="+", type=int, default=[1, 3], help="0-based unseen classes.")
    parser.add_argument("--samples_type", default="ratio", choices=["ratio", "same_num"])
    parser.add_argument("--train_ratio", type=float, default=0.05)
    parser.add_argument("--train_num", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", default=str(THIS_DIR / "Records"))
    parser.add_argument("--disable_progress", action="store_true")
    return parser.parse_args()


def default_epochs(method: str) -> int:
    return {
        "mft": 120,
        "ncglf2": 80,
        "exvit": 200,
        "frit": 80,
        "s2crossmamba": 120,
        "udm": 120,
        "unet": 120,
    }[method]


def build_method(method: str, bundle, patch_size: int, nclass: int):
    method = "s2crossmamba" if method == "s2mamba" else method
    primary_channels = bundle.primary.shape[2]
    secondary_channels = bundle.secondary.shape[2]
    fused_channels = bundle.fused.shape[2]

    if method == "mft":
        model = MFT(fm=16, num_hsi_bands=primary_channels, num_aux_bands=secondary_channels, num_classes=nclass, patch_size=patch_size)
        criterion = nn.CrossEntropyLoss()

        def train_loss_fn(batch, model, device):
            x1 = batch["primary"].to(device).flatten(2)
            x2 = batch["secondary"].to(device).flatten(2)
            labels = batch["label"].to(device)
            logits = model(x1, x2)
            return criterion(logits, labels), logits

        def eval_logits_fn(batch, model, device):
            return model(batch["primary"].to(device).flatten(2), batch["secondary"].to(device).flatten(2))

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return model, optimizer, scheduler, train_loss_fn, eval_logits_fn

    if method == "ncglf2":
        with temp_sys_path(THIS_DIR / "NCGLF2"):
            model_module = load_module("ncglf2_model_hzscm", THIS_DIR / "NCGLF2" / "model.py")
            utils_module = load_module("ncglf2_utils_hzscm", THIS_DIR / "NCGLF2" / "utils.py")
        model = model_module.NCGLF(num_classes=nclass, patch_size=patch_size, encoder_dim=64, depth=2, c1=primary_channels, c2=secondary_channels)
        criterion = utils_module.FocalLoss()

        def train_loss_fn(batch, model, device):
            labels = batch["label"].to(device)
            logits, con_loss = model(batch["fused"].to(device))
            return 0.2 * criterion(logits, labels) + 0.8 * con_loss, logits

        def eval_logits_fn(batch, model, device):
            return model(batch["fused"].to(device))[0]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.9)
        return model, optimizer, scheduler, train_loss_fn, eval_logits_fn

    if method == "exvit":
        exvit_module = load_module("exvit_model_hzscm", THIS_DIR / "ExViT" / "MViT_pytorch_upload.py")
        model = exvit_module.MViT(
            patch_size=patch_size,
            num_patches=[primary_channels, secondary_channels],
            num_classes=nclass,
            dim=64,
            depth=6,
            heads=4,
            mlp_dim=32,
            dropout=0.1,
            emb_dropout=0.1,
            mode="MViT",
        )
        criterion = nn.CrossEntropyLoss()

        def train_loss_fn(batch, model, device):
            labels = batch["label"].to(device)
            logits = model(batch["primary"].to(device), batch["secondary"].to(device))
            return criterion(logits, labels), logits

        def eval_logits_fn(batch, model, device):
            return model(batch["primary"].to(device), batch["secondary"].to(device))

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        return model, optimizer, scheduler, train_loss_fn, eval_logits_fn

    if method == "frit":
        model_module = load_module("frit_model_hzscm", THIS_DIR / "Fractional-Fourier-Image-Transformer" / "modelF.py")
        net = model_module.SSRN_network(fused_channels, nclass)
        vit = model_module.ViT(
            image_size=patch_size,
            patch_size=1,
            num_classes=nclass,
            dim=1024,
            depth=6,
            mlp_dim=2048,
            channels=24,
            dropout=0.1,
            emb_dropout=0.1,
        )
        criterion = nn.CrossEntropyLoss()

        class FrITWrapper(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                return self.head(self.backbone(x))

        model = FrITWrapper(net, vit)

        def train_loss_fn(batch, model, device):
            labels = batch["label"].to(device)
            x = batch["fused"].permute(0, 2, 3, 1).unsqueeze(1).to(device)
            logits = model(x)
            return criterion(logits, labels), logits

        def eval_logits_fn(batch, model, device):
            x = batch["fused"].permute(0, 2, 3, 1).unsqueeze(1).to(device)
            return model(x)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
        return model, optimizer, scheduler, train_loss_fn, eval_logits_fn

    if method == "s2crossmamba":
        with temp_sys_path(THIS_DIR / "S2CrossMamba" / "S2CrossMamba"):
            model_module = load_module("s2crossmamba_model_hzscm", THIS_DIR / "S2CrossMamba" / "S2CrossMamba" / "S2CrossMamba.py")
        model = model_module.S2CrossMamba(
            num_classes=nclass,
            AuHu=(secondary_channels > 1),
            Lidar_c=secondary_channels,
        )
        criterion = nn.CrossEntropyLoss()

        def train_loss_fn(batch, model, device):
            labels = batch["label"].to(device)
            x1 = batch["primary"].unsqueeze(1).to(device)
            x2 = batch["secondary"].to(device)
            logits = model(x1, x2)
            return criterion(logits, labels), logits

        def eval_logits_fn(batch, model, device):
            return model(batch["primary"].unsqueeze(1).to(device), batch["secondary"].to(device))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return model, optimizer, scheduler, train_loss_fn, eval_logits_fn

    if method == "udm":
        with temp_sys_path(THIS_DIR / "UDM"):
            model_module = load_module("udm_model_hzscm", THIS_DIR / "UDM" / "model.py")
            loss_module = load_module("udm_loss_hzscm", THIS_DIR / "UDM" / "loss.py")
        args = SimpleNamespace(
            num_classes=nclass,
            hsi_channel=primary_channels,
            lidar_channel=secondary_channels,
            img_size=patch_size,
            patch_size=3,
            dim=256 if bundle.dataset_name in {"Houston", "MUUFL"} else 128,
            depth=2,
            hidden_c=128,
            hidden_s=128,
            is_cls_token=False,
            in_channels=128,
            mlp_head="None",
        )
        model = model_module.UDM(args)

        def train_loss_fn(batch, model, device):
            x1 = batch["primary"].to(device)
            x2 = batch["secondary"].to(device)
            labels = batch["label"].to(device)
            outputs = model(x1, x2)
            fusion_logits, hsi_logits, lidar_logits, hsi_mu, hsi_logvar, lidar_mu, lidar_logvar, mu, logvar, z, _ = outputs
            conloss = loss_module.con_loss(hsi_mu, torch.exp(hsi_logvar), lidar_mu, torch.exp(lidar_logvar))
            loss = loss_module.totalloss(
                fusion_logits,
                hsi_logits,
                labels,
                lidar_logits,
                hsi_mu,
                hsi_logvar,
                lidar_mu,
                lidar_logvar,
                mu,
                logvar,
                z,
            )
            loss = loss + 5e-4 * loss_module.KL_cross(hsi_mu, hsi_logvar, lidar_mu, lidar_logvar) + conloss * 1e-3
            return loss, fusion_logits

        def eval_logits_fn(batch, model, device):
            return model(batch["primary"].to(device), batch["secondary"].to(device))[0]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
        return model, optimizer, scheduler, train_loss_fn, eval_logits_fn

    if method == "unet":
        with temp_sys_path(THIS_DIR / "UNet"):
            from unet import UNet  # type: ignore
        base_model = UNet(n_channels=fused_channels, n_classes=nclass, bilinear=False)
        criterion = nn.CrossEntropyLoss()

        class CenterPixelUNet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                logits = self.model(x)
                center = x.shape[-1] // 2
                return logits[:, :, center, center]

        model = CenterPixelUNet(base_model)

        def train_loss_fn(batch, model, device):
            labels = batch["label"].to(device)
            logits = model(batch["fused"].to(device))
            return criterion(logits, labels), logits

        def eval_logits_fn(batch, model, device):
            return model(batch["fused"].to(device))

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return model, optimizer, scheduler, train_loss_fn, eval_logits_fn

    raise ValueError(f"Unknown method: {method}")


def run_epoch(model, loader, optimizer, train_loss_fn, device, epoch_idx=None, total_epochs=None, disable_progress=False):
    model.train()
    total_loss = 0.0
    total = 0
    desc = "Train"
    if epoch_idx is not None and total_epochs is not None:
        desc = f"Epoch {epoch_idx + 1}/{total_epochs}"
    iterator = loader if disable_progress or tqdm is None else tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in iterator:
        optimizer.zero_grad()
        loss, logits = train_loss_fn(batch, model, device)
        loss.backward()
        optimizer.step()
        batch_size = logits.shape[0]
        total_loss += loss.item() * batch_size
        total += batch_size
        if not disable_progress and tqdm is not None:
            iterator.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(total, 1)


@torch.no_grad()
def predict_logits(model, loader, eval_logits_fn, device, disable_progress=False):
    model.eval()
    all_logits = []
    all_labels = []
    iterator = loader if disable_progress or tqdm is None else tqdm(loader, desc="Test", leave=False, dynamic_ncols=True)
    for batch in iterator:
        logits = eval_logits_fn(batch, model, device)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_logits, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def run_single_repeat(args, repeat_idx):
    method = "s2crossmamba" if args.method == "s2mamba" else args.method
    seed = args.seed + repeat_idx
    set_seed(seed)
    bundle = load_dataset(args.dataset, args.data_root)
    split = build_known_train_full_test_split(
        bundle.gt,
        del_class=args.del_class,
        seed=seed,
        samples_type=args.samples_type,
        train_ratio=args.train_ratio,
        train_num=args.train_num,
        val_ratio=args.val_ratio,
    )

    patch_size = args.patch_size or DEFAULT_METHOD_PATCH[method]
    nclass = int(bundle.gt.max())
    primary_cube = bundle.primary
    if method == "s2crossmamba":
        primary_cube = project_cube_pca(primary_cube, out_channels=30)
    fused_cube = np.concatenate([primary_cube, bundle.secondary], axis=2).astype(np.float32)
    train_set = MultiModalPatchDataset(primary_cube, bundle.secondary, fused_cube, split.train_coords, split.train_labels, patch_size)
    test_set = MultiModalPatchDataset(primary_cube, bundle.secondary, fused_cube, split.test_coords, split.test_labels, patch_size)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    method_bundle = SimpleNamespace(
        dataset_name=bundle.dataset_name,
        primary=primary_cube,
        secondary=bundle.secondary,
        fused=fused_cube,
        gt=bundle.gt,
        class_names=bundle.class_names,
    )
    model, optimizer, scheduler, train_loss_fn, eval_logits_fn = build_method(method, method_bundle, patch_size, nclass)
    device = torch.device(args.device)
    model = model.to(device)

    epochs = args.epochs or default_epochs(args.method)
    print(
        f"\n[Repeat {repeat_idx + 1}/{args.repeats}] "
        f"method={method} dataset={args.dataset} patch={patch_size} "
        f"train={len(train_set)} test={len(test_set)} unseen={args.del_class}"
    )
    tic_train = time.time()
    epoch_iter = range(epochs)
    if not args.disable_progress and tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc=f"Repeat {repeat_idx + 1} Epochs", leave=True, dynamic_ncols=True)
    for epoch_idx in epoch_iter:
        epoch_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            train_loss_fn,
            device,
            epoch_idx=epoch_idx,
            total_epochs=epochs,
            disable_progress=args.disable_progress,
        )
        if scheduler is not None:
            scheduler.step()
        if not args.disable_progress and tqdm is not None:
            epoch_iter.set_postfix(train_loss=f"{epoch_loss:.4f}")
    train_time = time.time() - tic_train

    tic_test = time.time()
    logits, labels = predict_logits(model, test_loader, eval_logits_fn, device, disable_progress=args.disable_progress)
    pred = logits.argmax(axis=1)
    test_time = time.time() - tic_test

    metrics = compute_metrics(labels, pred, nclass)
    su_metrics = compute_seen_unseen_metrics(metrics["per_class_acc"], bundle.gt, args.del_class)
    result = {
        "oa": metrics["oa"],
        "aa": metrics["aa"],
        "kappa": metrics["kappa"],
        "s_oa": su_metrics["s_oa"],
        "u_oa": su_metrics["u_oa"],
        "h_oa": su_metrics["h_oa"],
        "s_aa": su_metrics["s_aa"],
        "u_aa": su_metrics["u_aa"],
        "h_aa": su_metrics["h_aa"],
        "train_time": train_time,
        "test_time": test_time,
    }
    print(
        f"[Repeat {repeat_idx + 1}/{args.repeats}] "
        f"OA={result['oa']:.4f} AA={result['aa']:.4f} Kappa={result['kappa']:.4f} "
        f"S_OA={result['s_oa']:.4f} U_OA={result['u_oa']:.4f} H_OA={result['h_oa']:.4f}"
    )
    return result


def main():
    args = parse_args()
    rows = []
    results = []
    print(
        f"Start baseline run: method={args.method}, dataset={args.dataset}, "
        f"repeats={args.repeats}, device={args.device}, unseen={args.del_class}"
    )
    repeat_iter = range(args.repeats)
    if not args.disable_progress and tqdm is not None:
        repeat_iter = tqdm(repeat_iter, desc="Repeats", leave=True, dynamic_ncols=True)
    for repeat_idx in repeat_iter:
        result = run_single_repeat(args, repeat_idx)
        results.append(result)
        if not args.disable_progress and tqdm is not None:
            repeat_iter.set_postfix(last_oa=f"{result['oa']:.4f}", last_h=f"{result['h_oa']:.4f}")

    agg = aggregate_results(results)
    row = {
        "method": "s2crossmamba" if args.method == "s2mamba" else args.method,
        "dataset": args.dataset,
        "unseen_idx": ";".join(str(i) for i in args.del_class),
        **{key: f"{value:.6f}" for key, value in agg.items()},
    }
    rows.append(row)
    save_path = Path(args.save_dir) / f"{args.method}_{args.dataset}_del_{'-'.join(str(i) for i in args.del_class)}.csv"
    save_csv(rows, save_path)
    print(f"\nSaved results to {save_path}")
    for key, value in agg.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
