# -*- coding: utf-8 -*-
"""
主实验入口脚本。

整体流程分成两阶段：
1. 先只用真实标签对分割网络做一个短暂预训练；
2. 利用预训练结果修正伪标签，再用"真实标签 + 修正后的伪标签"继续训练。

阅读这份脚本时，可以先记住几组最重要的变量：
- `Train_Split_Data / Test_Split_Data`：切成 patch 后的输入图像
- `Train_Split_GT / Test_Split_GT`：与 patch 对齐的真实标签
- `Train_Split_PL / Train_Split_PL_C`：原始伪标签 / 修正后的伪标签
- `Train_Label / Test_Label`：整图尺度的训练标签 / 测试标签
- `PreOutputWhole / OutputWhole`：模型预测重新拼回整图后的结果
- `AC / OA / AA / KA`：每次实验保存的精度指标
"""

import time
import datetime
import csv
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import load_args
from data_read import read_data

from get_map import generate
from hyper_dataset import HyperData
from util import PatchStack, Kappa, ClassificationAccuracy, SpiltHSI, get_USH, correct_pseudo_labels

from seg_model import HSI_Seg_Hard
from data_read_singlemodal import DATASET_CLASS_NAMES


# 数据集类别数
DATASET_NCLASS = {
    'Houston': 15,
    'HS-SAR-Berlin': 8,
    'MUUFL': 11,
}

# 早停观察窗口
LOSS_NUM = 4


def run_single_repeat(args, del_class, repeat_idx, device):
    """执行一次完整的两阶段实验，返回指标字典。

    参数:
        args: 实验配置
        del_class: unseen 类别列表 (0-based)
        repeat_idx: 当前重复编号
        device: 计算设备

    返回:
        dict: 包含所有指标的字典
    """
    # 读取数据、伪标签和 patch 切分结果
    Train_Split_Data, Train_Split_GT, Train_Split_PL, \
    Test_Split_Data, Test_Split_GT, Test_Split_PL, \
    patch_height, patch_width, data, gt, sp_gt, pseudo_labels, \
    Train_Label, Test_Label = read_data(args, repeat_idx)

    if del_class is not None:
        mask = np.zeros_like(Train_Split_GT)
        mask_g = np.zeros_like(Train_Label)
        for value in del_class:
            mask[Train_Split_GT == value + 1] = 1
            mask_g[Train_Label == value + 1] = 1
        Train_Split_GT *= (1 - mask)
        Train_Label_U = Train_Label * (1 - mask_g)
    else:
        Train_Label_U = Train_Label

    nband = Train_Split_Data.shape[-1]
    nclass = np.max(gt)

    model = HSI_Seg_Hard(nband, args.hidden_dim, args.out_dim, nclass).to(device)

    ############################################################################### 第一阶段：仅用真实标签做预训练
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 160, 240], gamma=0.5, last_epoch=-1)

    train_h = HyperData(
        (np.transpose(Train_Split_Data, (0, 3, 1, 2)).astype("float32"),
         Train_Split_GT, Train_Split_PL), None)
    test_h = HyperData(
        (np.transpose(Test_Split_Data, (0, 3, 1, 2)).astype("float32"),
         Test_Split_GT, Test_Split_PL), None)
    trainloader = DataLoader(train_h, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_h, batch_size=1, shuffle=False)

    loss_j = []
    tic1 = time.time()
    for epoch in range(args.pre_epoch):
        model.train()
        total_loss = 0
        for idx, (inputs, labels, pseudo_label) in enumerate(trainloader):
            x = inputs.float().to(device)
            labels, pseudo_label = labels.to(device), pseudo_label.to(device)
            _, loss_l = model.pre_forward(x, labels.long())
            loss = loss_l
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item() * inputs.shape[0]
        scheduler.step()
        total_loss = total_loss / Train_Split_Data.shape[0]
        loss_j.append(total_loss)
        if (epoch + 1) % 10 == 0:
            print(f'  [pre] epoch: {epoch}, loss: {total_loss:.4f}')
        if np.sum(np.array(loss_j)[-1 - LOSS_NUM:-1] < 1e-5) == LOSS_NUM:
            break
    toc1 = time.time()
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, './pretrain_net.pt')

    # 预训练模型测试
    tic2 = time.time()
    model.eval()
    Output = []
    for Testbatch_idx, (Testinputs, Testtargets, _) in enumerate(testloader):
        Testinputs, Testtargets = Testinputs.to(device), Testtargets.to(device)
        _, Testoutput = model.test(Testinputs)
        Testoutput = Testoutput.data.cpu().numpy()
        Output.append(Testoutput[0])
    PreOutputWhole = PatchStack(Output, data.shape[0], data.shape[1], patch_height, patch_width,
                                args.split_height, args.split_width, args.EDGE, nclass)
    toc2 = time.time()

    ac, oa, aa, rightNum, testNum = ClassificationAccuracy(PreOutputWhole + 1, Test_Label, nclass)
    kappa, cm = Kappa(PreOutputWhole + 1, Test_Label, nclass)
    print(f"  [pre] OA={oa:.4f}  AA={aa:.4f}  Kappa={kappa:.4f}")

    pre_metrics = {
        'pre_oa': oa, 'pre_aa': aa, 'pre_kappa': kappa,
        'pre_ac': ac, 'pre_cm': cm,
        'pre_train_time': toc1 - tic1, 'pre_test_time': toc2 - tic2,
    }

    del model, Testinputs, Testtargets, Testoutput, optimizer, scheduler, trainloader
    torch.cuda.empty_cache()

    ############################################################################### 修正伪标签
    scores, pseudo_labels_c, pseudo_probs_c = correct_pseudo_labels(
        data, gt, sp_gt, Train_Label_U, pseudo_labels, PreOutputWhole, del_class,
        n_neighbours=args.n_neighbours, n_clusters=100, tao=args.tao, threshold=args.threshold)
    ac_c, oa_c, aa_c, _, _ = ClassificationAccuracy(pseudo_labels_c + 1, Test_Label, nclass)
    kappa_c, cm_c = Kappa(pseudo_labels_c + 1, Test_Label, nclass)
    print(f"  [corr] OA={oa_c:.4f}  AA={aa_c:.4f}  Kappa={kappa_c:.4f}")

    ################################################################################# 第二阶段：用修正后的伪标签继续训练
    _, Train_Split_PL_C, _ = SpiltHSI(data, pseudo_labels_c, pseudo_probs_c,
                                       [args.split_height, args.split_width], args.EDGE)

    model = HSI_Seg_Hard(nband, args.hidden_dim, args.out_dim, nclass).to(device)
    train_h = HyperData(
        (np.transpose(Train_Split_Data, (0, 3, 1, 2)).astype("float32"),
         Train_Split_GT, Train_Split_PL_C), None)
    trainloader = DataLoader(train_h, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200, 300], gamma=0.5, last_epoch=-1)

    loss_j = []
    loss_ju = []
    tic1 = time.time()
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        u_loss = 0
        for idx, (inputs, labels, pseudo_label) in enumerate(trainloader):
            x = inputs.float().to(device)
            labels, pseudo_label = labels.to(device), pseudo_label.to(device)
            out, loss_l, loss_u = model(x, pseudo_label, labels.long())
            loss = loss_l + args.beta * loss_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item() * inputs.shape[0]
            u_loss = u_loss + loss_u.item() * inputs.shape[0]
        scheduler.step()
        total_loss = total_loss / Train_Split_Data.shape[0]
        u_loss = u_loss / Train_Split_Data.shape[0]
        loss_j.append(total_loss)
        loss_ju.append(u_loss)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        if (epoch + 1) % 10 == 0:
            print(f'  [main] epoch: {epoch}, loss: {total_loss:.4f}, u_loss: {u_loss:.4f}')
        if np.sum(np.array(loss_ju)[-LOSS_NUM:] < 0.05) == LOSS_NUM and epoch >= 50:
            break
    toc1 = time.time()

    ################################################################################# 最终测试
    model.eval()
    Output = []
    tic2 = time.time()
    for Testbatch_idx, (Testinputs, Testtargets, _) in enumerate(testloader):
        Testinputs, Testtargets = Testinputs.to(device), Testtargets.to(device)
        TestoutputFeat, Testoutput = model.test(Testinputs)
        Testoutput = Testoutput.data.cpu().numpy()
        Output.append(Testoutput[0])
    OutputWhole = PatchStack(Output, data.shape[0], data.shape[1], patch_height, patch_width,
                             args.split_height, args.split_width, args.EDGE, nclass)
    toc2 = time.time()

    ac, oa, aa, rightNum, testNum = ClassificationAccuracy(OutputWhole + 1, Test_Label, nclass)
    kappa, cm = Kappa(OutputWhole + 1, Test_Label, nclass)

    ys_oa, yu_oa, yh_oa = get_USH(ac, [np.sum(Test_Label == i + 1) for i in range(nclass)], del_class, metric='OA')
    ys_aa, yu_aa, yh_aa = get_USH(ac, [np.sum(Test_Label == i + 1) for i in range(nclass)], del_class, metric='AA')

    print(f"  [final] OA={oa:.4f}  AA={aa:.4f}  Kappa={kappa:.4f}")
    print(f"  [GZSL] S_OA={ys_oa:.4f}  U_OA={yu_oa:.4f}  H_OA={yh_oa:.4f}  |  "
          f"S_AA={ys_aa:.4f}  U_AA={yu_aa:.4f}  H_AA={yh_aa:.4f}")

    # 保存分类可视化图
    day_str = datetime.datetime.now().strftime('%m_%d_%H_%M')
    del_str = 'del_class_' + (''.join(str(d) for d in del_class) if del_class else 'None')
    generate(OutputWhole, './ResultsImage/' + args.dataset + '_' + str(day_str) + '_'
             + str(args.curr_train_ratio) + '_' + del_str + '_' + str(repeat_idx) + '_'
             + "{:.2f}".format(oa * 100))

    final_metrics = {
        'nclass': nclass,
        'oa': oa, 'aa': aa, 'kappa': kappa,
        'ac': ac, 'cm': cm,
        's_oa': ys_oa, 'u_oa': yu_oa, 'h_oa': yh_oa,
        's_aa': ys_aa, 'u_aa': yu_aa, 'h_aa': yh_aa,
        'train_time': toc1 - tic1, 'test_time': toc2 - tic2,
    }

    del model, Testinputs, Testtargets, Testoutput, optimizer, scheduler, trainloader, testloader
    torch.cuda.empty_cache()

    return {**pre_metrics, **final_metrics}


def run_experiment_group(args, del_class, num_repeats=None, device=None):
    """多次重复实验并聚合结果，返回一行 CSV 数据。

    参数:
        args: 实验配置
        del_class: unseen 类别列表 (0-based)
        num_repeats: 重复次数，默认使用 args 中的设定
        device: 计算设备

    返回:
        dict: 扁平化的聚合指标（均值±标准差）
    """
    if num_repeats is None:
        num_repeats = 5
    if device is None:
        device = torch.device('cuda:0')

    dataset = args.dataset
    nclass = DATASET_NCLASS.get(dataset, 15)
    class_names = DATASET_CLASS_NAMES.get(dataset, [f'class_{i}' for i in range(nclass)])
    unseen_names = [class_names[i] for i in del_class] if del_class else []

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}  |  Unseen: {del_class} ({', '.join(unseen_names)})")
    print(f"  Repeats: {num_repeats}")
    print(f"{'='*60}")

    all_results = []
    for r in range(num_repeats):
        print(f"\n--- Repeat {r+1}/{num_repeats} ---")
        metrics = run_single_repeat(args, del_class, r, device)
        all_results.append(metrics)

    # 聚合
    agg = _aggregate(all_results)

    # 打印汇总
    _print_summary(agg, dataset, del_class, unseen_names, num_repeats)

    # 返回扁平化的一行数据
    row = _make_csv_row(agg, dataset, del_class, unseen_names, nclass)
    return row


def _aggregate(results):
    """将多次实验结果聚合为均值±标准差。"""
    keys_scalar = [
        'pre_oa', 'pre_aa', 'pre_kappa',
        'oa', 'aa', 'kappa',
        's_oa', 'u_oa', 'h_oa',
        's_aa', 'u_aa', 'h_aa',
        'train_time', 'test_time',
    ]
    agg = {}
    for key in keys_scalar:
        values = [r[key] for r in results]
        agg[f'{key}_mean'] = np.mean(values)
        agg[f'{key}_std'] = np.std(values)
    agg['nclass'] = results[0]['nclass']
    return agg


def _print_summary(agg, dataset, del_class, unseen_names, num_repeats):
    """打印格式化汇总表。"""
    unseen_str = f"{del_class} ({', '.join(unseen_names)})" if unseen_names else str(del_class)

    def fmt(key):
        return f"{agg[key+'_mean']:.4f} ± {agg[key+'_std']:.4f}"

    print(f"\n{'='*60}")
    print(f"  Experiment Summary")
    print(f"{'='*60}")
    print(f"  Dataset:   {dataset}")
    print(f"  Unseen:    {unseen_str}")
    print(f"  Repeats:   {num_repeats}")
    print(f"{'-'*60}")
    print(f"  {'':12s} {'OA':>20s} {'AA':>20s} {'Kappa':>20s}")
    print(f"  {'Pre-train':12s} {fmt('pre_oa'):>20s} {fmt('pre_aa'):>20s} {fmt('pre_kappa'):>20s}")
    print(f"  {'Final':12s} {fmt('oa'):>20s} {fmt('aa'):>20s} {fmt('kappa'):>20s}")
    print(f"{'-'*60}")
    print(f"  GZSL (OA)   S={fmt('s_oa')}   U={fmt('u_oa')}   H={fmt('h_oa')}")
    print(f"  GZSL (AA)   S={fmt('s_aa')}   U={fmt('u_aa')}   H={fmt('h_aa')}")
    print(f"{'-'*60}")
    print(f"  Avg train time: {agg['train_time_mean']:.1f}s   Avg test time: {agg['test_time_mean']:.1f}s")
    print(f"{'='*60}\n")


CSV_COLUMNS = [
    'dataset', 'unseen_idx', 'unseen_names', 'nclass',
    'pre_OA_mean', 'pre_OA_std', 'pre_AA_mean', 'pre_AA_std', 'pre_Kappa_mean', 'pre_Kappa_std',
    'OA_mean', 'OA_std', 'AA_mean', 'AA_std', 'Kappa_mean', 'Kappa_std',
    'S_OA_mean', 'S_OA_std', 'U_OA_mean', 'U_OA_std', 'H_OA_mean', 'H_OA_std',
    'S_AA_mean', 'S_AA_std', 'U_AA_mean', 'U_AA_std', 'H_AA_mean', 'H_AA_std',
    'avg_train_time', 'avg_test_time',
]


def _make_csv_row(agg, dataset, del_class, unseen_names, nclass):
    """将聚合指标转为 CSV 行字典。"""
    row = {
        'dataset': dataset,
        'unseen_idx': ';'.join(str(d) for d in del_class),
        'unseen_names': ';'.join(unseen_names),
        'nclass': nclass,
        'pre_OA_mean': f"{agg['pre_oa_mean']:.6f}",
        'pre_OA_std': f"{agg['pre_oa_std']:.6f}",
        'pre_AA_mean': f"{agg['pre_aa_mean']:.6f}",
        'pre_AA_std': f"{agg['pre_aa_std']:.6f}",
        'pre_Kappa_mean': f"{agg['pre_kappa_mean']:.6f}",
        'pre_Kappa_std': f"{agg['pre_kappa_std']:.6f}",
        'OA_mean': f"{agg['oa_mean']:.6f}",
        'OA_std': f"{agg['oa_std']:.6f}",
        'AA_mean': f"{agg['aa_mean']:.6f}",
        'AA_std': f"{agg['aa_std']:.6f}",
        'Kappa_mean': f"{agg['kappa_mean']:.6f}",
        'Kappa_std': f"{agg['kappa_std']:.6f}",
        'S_OA_mean': f"{agg['s_oa_mean']:.6f}",
        'S_OA_std': f"{agg['s_oa_std']:.6f}",
        'U_OA_mean': f"{agg['u_oa_mean']:.6f}",
        'U_OA_std': f"{agg['u_oa_std']:.6f}",
        'H_OA_mean': f"{agg['h_oa_mean']:.6f}",
        'H_OA_std': f"{agg['h_oa_std']:.6f}",
        'S_AA_mean': f"{agg['s_aa_mean']:.6f}",
        'S_AA_std': f"{agg['s_aa_std']:.6f}",
        'U_AA_mean': f"{agg['u_aa_mean']:.6f}",
        'U_AA_std': f"{agg['u_aa_std']:.6f}",
        'H_AA_mean': f"{agg['h_aa_mean']:.6f}",
        'H_AA_std': f"{agg['h_aa_std']:.6f}",
        'avg_train_time': f"{agg['train_time_mean']:.2f}",
        'avg_test_time': f"{agg['test_time_mean']:.2f}",
    }
    return row


def save_results_csv(rows, path, append=False):
    """将结果行列表写入 CSV 文件。

    参数:
        rows: dict 列表，每个 dict 是一行数据
        path: CSV 文件路径
        append: 是否追加模式（默认覆盖）
    """
    import os
    mode = 'a' if append and os.path.exists(path) else 'w'
    write_header = not (append and os.path.exists(path) and os.path.getsize(path) > 0)

    with open(path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# 保留原有的命令行入口
if __name__ == '__main__':
    args = load_args()
    del_class = args.del_class

    row = run_experiment_group(args, del_class, num_repeats=5)

    # 保存 CSV
    day_str = datetime.datetime.now().strftime('%m_%d_%H_%M')
    del_str = 'del_class_' + (''.join(str(d) for d in del_class) if del_class else 'None')
    csv_path = f'Records/{args.dataset}/{args.dataset}_{day_str}_{del_str}.csv'
    import os
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    save_results_csv([row], csv_path)
    print(f"Results saved to {csv_path}")
