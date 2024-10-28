import argparse
import sys
sys.path.append("..")

import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import os
import shutil
import logging
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.chdir('/database/wuyonghuang/ShaSpec/')

import os.path as osp
from DualNet_SS import DualNet_SS as DualNet
from BraTSDataSet import BraTSDataSet, BraTSValDataSet, my_collate
import timeit
from tensorboardX import SummaryWriter
import loss_Dual as loss
from engine import Engine
from math import ceil

from DualNet_SS import conv3x3x3
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
from sendemail import let_me_know
from datetime import datetime

start = timeit.default_timer()
alpha = 0.1  # shared domain loss weight
beta = 0.02  # specific domain loss weight
calc_flops = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Shared-Specific model for 3D Medical Image Segmentation.")

    parser.add_argument("--data_dir", type=str, default='./datalist/')
    parser.add_argument("--train_list", type=str, default='BraTS20/BraTS20_train.csv')
    parser.add_argument("--val_list", type=str, default='BraTS20_val.csv')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/example/')
    parser.add_argument("--reload_path", type=str, default='snapshots/example/last.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=False)
    parser.add_argument("--random_scale", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=999)

    parser.add_argument("--norm_cfg", type=str, default='IN')  # normalization
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')  # activation
    parser.add_argument("--train_only", action="store_true")

    # 增加的参数
    parser.add_argument("--mode", type=str, default='0,1,2,3')  # 设置全模态
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--data_setting", type=str, default='m3ae', help='data setting: m3ae or shaspec')
    return parser


def assign_params(args):
    # Directly assign the parameters
    args.snapshot_dir = 'evaluations/debug/'
    args.input_size = '80, 160, 160'
    args.batch_size = 2
    args.num_gpus = 1
    args.num_steps = 10000
    args.val_pred_every = 100
    args.learning_rate = 1e-4
    args.num_classes = 3
    args.num_workers = 4
    args.train_list = 'BraTS18/m3ae_train3_new.csv'
    args.val_list = 'BraTS18/m3ae_val3_new.csv'
    args.random_mirror = True
    args.random_scale = True
    args.weight_std = True
    args.train_only = True
    args.reload_path = '/database/wuyonghuang/ShaSpec/snapshots/BraTS18_ShaSpec_[80,160,160]_SGD_b1_lr-2_alpha.1_beta.02_trainOnly/final.pth'
    args.reload_from_checkpoint = True
    
    return args

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2*num / den

    return dice.mean()


def compute_dice_score(preds, labels):

    preds = F.sigmoid(preds)

    pred_ET = preds[:, 0, :, :, :]
    pred_WT = preds[:, 1, :, :, :]
    pred_TC = preds[:, 2, :, :, :]
    label_ET = labels[:, 0, :, :, :]
    label_WT = labels[:, 1, :, :, :]
    label_TC = labels[:, 2, :, :, :]
    dice_ET = dice_score(pred_ET, label_ET).cpu().data.numpy()
    dice_WT = dice_score(pred_WT, label_WT).cpu().data.numpy()
    dice_TC = dice_score(pred_TC, label_TC).cpu().data.numpy()
    return dice_ET, dice_WT, dice_TC


def predict_sliding(args, net, imagelist, tile_size, classes):
    image, image_res = imagelist
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    full_probs = torch.from_numpy(full_probs).cuda()
    count_predictions = torch.from_numpy(count_predictions).cuda()

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]

                prediction, _, _, _ = net(img, val=True, mode='0,1,2,3')    # 这里应该是 args.mode 还是 直接 "0,1,2,3"

                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs


def validate(args, input_size, model, ValLoader, num_classes):
    # start to validate
    val_ET = 0.0
    val_WT = 0.0
    val_TC = 0.0

    for index, batch in enumerate(ValLoader):
        # print('%d processd'%(index))
        image, image_res, label, size, name, affine = batch
        image = image.cuda()
        image_res = image_res.cuda()
        label = label.cuda()
        with torch.no_grad():
            pred = predict_sliding(args, model, [image, image_res], input_size, num_classes)
            dice_ET, dice_WT, dice_TC = compute_dice_score(pred, label)
            val_ET += dice_ET
            val_WT += dice_WT
            val_TC += dice_TC

    return val_ET/(index+1), val_WT/(index+1), val_TC/(index+1)


def main():
    """Create the ConResNet model and then start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        # args = assign_params(args)    # 这里仅仅用于调试
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        code_dir = os.path.join(args.snapshot_dir, 'code')
        if not os.path.exists(code_dir):
            ignore_patterns = ['__pycache__', 'apex', 'data', 'snapshots', 'pre_snapshots', 'logs', 'wandb', 'evaluations']
            shutil.copytree('../ShaSpec', code_dir, ignore=shutil.ignore_patterns(*ignore_patterns))

        # 暂存，用于恢复
        temp = sys.stdout
        # 把输出重定向到文件
        f = open(os.path.join(args.snapshot_dir, 'output.log'), 'w')
        # 之后使用print函数，都将内容打印到 screenshot.log 文件中
        sys.stdout = f

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = DualNet(args=args, norm_cfg=args.norm_cfg, activation_cfg=args.activation_cfg,
                        num_classes=args.num_classes, weight_std=args.weight_std, self_att=False, cross_att=False)

        if calc_flops:
            from thop import profile
            input = torch.randn(1, 4, 80, 160, 160)
            macs, params = profile(model, inputs=(input,))
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            return

        model.train()
        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        # load checkpoint...
        print(f'是否重载模型: {args.reload_from_checkpoint}')
        reload_best_val_dice = None
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                # model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
                checkpoint = torch.load(args.reload_path)
                model = checkpoint['model']
                optimizer = checkpoint['optimizer']
                args.start_iters = checkpoint['iter']
                try:
                    reload_best_val_dice = checkpoint['best_val_dice']
                except:
                    print('加载的权重中没有 reload_best_val_dice. ')
                print("Loaded model trained for", args.start_iters, "iters")
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))
                exit(0)

        print('current mode:', args.mode)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        trainloader, train_sampler = engine.get_train_loader(BraTSDataSet(args.data_dir, args.train_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror), collate_fn=my_collate)
        valloader, val_sampler = engine.get_test_loader(BraTSValDataSet(args.data_dir, args.val_list))
        print('已加载 训练集和验证集！')

        if args.data_setting == 'm3ae':
            testloader, test_sampler = engine.get_test_loader(BraTSValDataSet(args.data_dir, "BraTS18/m3ae_test3_new.csv"))
            print('已加载 测试集！')

        f.flush()
        if args.data_setting == 'm3ae':
            print('testing ...')
            model.eval()

            test_ET, test_WT, test_TC = validate(args, input_size, model, testloader, args.num_classes) # todo: 在不同的缺失模态下进行 test. 
            if (args.local_rank == 0):
                print('{}, Testing iter = {}, ET = {:.2}, WT = {:.2}, TC = {:.2}'.format(str(datetime.now()), 0, test_ET, test_WT, test_TC))
        else:
            print('check your data setting!')

    end = timeit.default_timer()
    print(end - start, 'seconds')
    f.close()
    let_me_know(f'The phase TEST is finished!')


if __name__ == '__main__':
    main()
