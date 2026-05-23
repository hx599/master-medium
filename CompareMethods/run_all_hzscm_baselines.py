import argparse
import subprocess
import sys
from pathlib import Path
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


METHODS = ["mft", "ncglf2", "exvit", "frit", "s2crossmamba", "udm", "unet"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run all adapted baselines under the HZSCM hidden-class protocol.")
    parser.add_argument("--dataset", default="MUUFL", choices=["Houston", "HS-SAR-Berlin", "MUUFL"])
    parser.add_argument("--data_root", default="D:/Master_medium/dataset")
    parser.add_argument("--del_class", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--samples_type", default="ratio", choices=["ratio", "same_num"])
    parser.add_argument("--train_ratio", type=float, default=0.05)
    parser.add_argument("--train_num", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", default=str(Path(__file__).resolve().parent / "Records"))
    parser.add_argument("--disable_progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    script = Path(__file__).resolve().parent / "run_hzscm_baseline.py"
    method_iter = METHODS
    if not args.disable_progress and tqdm is not None:
        method_iter = tqdm(METHODS, desc="Methods", leave=True, dynamic_ncols=True)
    for method in method_iter:
        cmd = [
            sys.executable,
            str(script),
            "--method",
            method,
            "--dataset",
            args.dataset,
            "--data_root",
            args.data_root,
            "--samples_type",
            args.samples_type,
            "--train_ratio",
            str(args.train_ratio),
            "--val_ratio",
            str(args.val_ratio),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--save_dir",
            args.save_dir,
            "--del_class",
            *[str(item) for item in args.del_class],
        ]
        if args.disable_progress:
            cmd.append("--disable_progress")
        if args.train_num is not None:
            cmd.extend(["--train_num", str(args.train_num)])
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        print(f"\nRunning {method} ...")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
