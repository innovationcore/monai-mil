import argparse
import collections.abc
import os
import shutil
import time
import json

import gdown
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist
from monai.data.wsi_reader import WSIReader
from monai.metrics import Cumulative, CumulativeAverage
from monai.networks.nets import milmodel
from monai.transforms import (
    Compose,
    GridPatchd,
    LoadImaged,
    MapTransform,
    RandFlipd,
    RandGridPatchd,
    RandRotate90d,
    ScaleIntensityRanged,
    SplitDimd,
    ToTensord,
)
import sklearn
from sklearn.metrics import cohen_kappa_score, roc_curve, precision_recall_fscore_support
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from clearml import Logger
from clearml import Task


def train_epoch(model, loader, optimizer, scaler, epoch, args):
    """One train epoch over the dataset"""

    model.train()
    criterion = nn.BCEWithLogitsLoss()

    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()

    start_time = time.time()
    loss, acc = 0.0, 0.0

    for idx, batch_data in enumerate(loader):

        data = batch_data["image"].as_subclass(torch.Tensor).cuda(args.rank)
        target = batch_data["label"].as_subclass(torch.Tensor).cuda(args.rank)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            logits = model(data)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = (logits.sigmoid().sum(1).detach().round() == target.sum(1).round()).float().mean()

        run_loss.append(loss)
        run_acc.append(acc)

        loss = run_loss.aggregate()
        acc = run_acc.aggregate()

        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.epochs, idx, len(loader)),
                "loss: {:.4f}".format(loss),
                "acc: {:.4f}".format(acc),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()

    return loss, acc


def val_epoch(model, loader, epoch, args, max_tiles=None):
    """One validation epoch over the dataset"""

    model.eval()

    model2 = model if not args.distributed else model.module
    has_extra_outputs = model2.mil_mode == "att_trans_pyramid"
    extra_outputs = model2.extra_outputs
    calc_head = model2.calc_head

    criterion = nn.BCEWithLogitsLoss()

    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()
    FILES = []  # TODO(avirodov): this is probably not good for multiprocessing.
    PROBS = Cumulative()
    PREDS = Cumulative()
    TARGETS = Cumulative()

    start_time = time.time()
    loss, acc = 0.0, 0.0

    with torch.no_grad():

        for idx, batch_data in enumerate(loader):

            data = batch_data["image"].as_subclass(torch.Tensor).cuda(args.rank)
            target = batch_data["label"].as_subclass(torch.Tensor).cuda(args.rank)

            with autocast(enabled=args.amp):

                if max_tiles is not None and data.shape[1] > max_tiles:
                    # During validation, we want to use all instances/patches
                    # and if its number is very big, we may run out of GPU memory
                    # in this case, we first iteratively go over subsets of patches to calculate backbone features
                    # and at the very end calculate the classification output

                    logits = []
                    logits2 = []

                    for i in range(int(np.ceil(data.shape[1] / float(max_tiles)))):
                        data_slice = data[:, i * max_tiles: (i + 1) * max_tiles]
                        logits_slice = model(data_slice, no_head=True)
                        logits.append(logits_slice)

                        if has_extra_outputs:
                            logits2.append(
                                [
                                    extra_outputs["layer1"],
                                    extra_outputs["layer2"],
                                    extra_outputs["layer3"],
                                    extra_outputs["layer4"],
                                ]
                            )

                    logits = torch.cat(logits, dim=1)
                    if has_extra_outputs:
                        extra_outputs["layer1"] = torch.cat([l[0] for l in logits2], dim=0)
                        extra_outputs["layer2"] = torch.cat([l[1] for l in logits2], dim=0)
                        extra_outputs["layer3"] = torch.cat([l[2] for l in logits2], dim=0)
                        extra_outputs["layer4"] = torch.cat([l[3] for l in logits2], dim=0)

                    logits = calc_head(logits)

                else:
                    # if number of instances is not big, we can run inference directly
                    logits = model(data)

                loss = criterion(logits, target)

            prob = logits.sigmoid().detach()
            pred = logits.sigmoid().sum(1).detach().round()
            target = target.sum(1).round()
            acc = (pred == target).float().mean()

            run_loss.append(loss)
            run_acc.append(acc)
            loss = run_loss.aggregate()
            acc = run_acc.aggregate()

            PROBS.extend(prob)
            PREDS.extend(pred)
            TARGETS.extend(target)
            FILES.extend(batch_data["image_name"])

            if args.rank == 0:
                print(
                    "Val epoch {}/{} {}/{}".format(epoch, args.epochs, idx, len(loader)),
                    "loss: {:.4f}".format(loss),
                    "acc: {:.4f}".format(acc),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

        # Calculate QWK metric (Quadratic Weigted Kappa) https://en.wikipedia.org/wiki/Cohen%27s_kappa
        # TODO(avirodov): PROBS should not be flattened if num_classes > 1 !
        PROBS = PROBS.get_buffer().cpu().numpy().flatten()
        PREDS = PREDS.get_buffer().cpu().numpy()
        TARGETS = TARGETS.get_buffer().cpu().numpy()
        qwk = cohen_kappa_score(PREDS.astype(np.float64), TARGETS.astype(np.float64), weights="quadratic")

        if args.num_classes == 1:
            fpr, tpr, thresholds = roc_curve(TARGETS, PROBS, pos_label=1)
            auc = sklearn.metrics.auc(fpr, tpr)
        else:
            # Those are not well-defined for non-binary classification (maybe they can be if you can threshold along
            #  a continuous parameter), and the network outputs probabilities only in case num_classes == 1. Otherwise
            #  the PROBS array contains some numbers that sum to the label integer (rounded), and cannot be interpreted
            #  as probability without changing the approach set by LabelEncodeIntegerGraded.
            fpr, tpr, thresholds = -1, -1, []
            auc = -1
        if args.rank == 0:
            print(f'{FILES=}')
            print(f'{PROBS=}')
            print(f'{PREDS=}')
            print(f'{TARGETS=}')
            print(f'{fpr=}')
            print(f'{tpr=}')
            print(f'{thresholds=}')

        if args.rank == 0:

            prediction_map = dict()
            prediction_file_path = 'predictions_val.json'
            if os.path.exists(prediction_file_path):

                if epoch == 0:
                    os.remove(prediction_file_path)
                else:
                    with open(prediction_file_path, 'r') as f:
                        prediction_map = json.load(f)

            if epoch not in prediction_map:
                prediction_map[epoch] = []

            for a_file, a_target, a_prob in zip(FILES, TARGETS.tolist(), PROBS.tolist()):
                pred_file = dict()
                pred_file['file_path'] = a_file
                pred_file['target'] = a_target
                pred_file['prediction'] = int(a_prob >= 0.5)
                pred_file['probability'] = a_prob
                if pred_file['target'] != pred_file['prediction']:
                    pred_file['correct'] = False
                else:
                    pred_file['correct'] = True
                prediction_map[epoch].append(pred_file)

            with open(prediction_file_path, 'w') as json_file:
                json.dump(prediction_map, json_file, indent=4)

            fp = open(os.path.join('.', f'predictions_val.csv'), 'w')
            fp.write('file,target,prediction,probability\n')
            for a_file, a_target, a_prob in zip(FILES, TARGETS.tolist(), PROBS.tolist()):
                fp.write(f'{a_file},{a_target},{int(a_prob >= 0.5)},{a_prob}\n')
            fp.close()

        precision, recall, fbeta_score, support = precision_recall_fscore_support(TARGETS, PREDS, pos_label=1,
                                                                                  average="weighted")

    # return loss, acc, qwk
    return loss, acc, qwk, fpr, tpr, auc, precision, recall, fbeta_score


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0):
    """Save checkpoint"""

    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()

    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}

    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


class LabelEncodeIntegerGraded(MapTransform):
    """
    Convert an integer label to encoded array representation of length num_classes,
    with 1 filled in up to label index, and 0 otherwise. For example for num_classes=5,
    embedding of 2 -> (1,1,0,0,0)

    Args:
        num_classes: the number of classes to convert to encoded format.
        keys: keys of the corresponding items to be transformed. Defaults to ``'label'``.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
            self,
            num_classes: int,
            keys: KeysCollection = "label",
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            label = int(d[key])

            lz = np.zeros(self.num_classes, dtype=np.float32)
            lz[:label] = 1.0
            # alternative oneliner lz=(np.arange(self.num_classes)<int(label)).astype(np.float32) #same oneliner
            d[key] = lz

        return d


def list_data_collate(batch: collections.abc.Sequence):
    """
    Combine instances from a list of dicts into a single dict, by stacking them along first dim
    [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
    followed by the default collate which will form a batch BxNx3xHxW
    """

    for i, item in enumerate(batch):
        image_tensor = torch.stack([ix["image"] for ix in item], dim=0)
        patch_location_tensor = torch.tensor(item[0]["image"].meta["location"])
        data = item[0]
        data["image"] = image_tensor
        data["patch_location"] = patch_location_tensor
        batch[i] = data
    return default_collate(batch)


def main_worker(gpu, args):

    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * torch.cuda.device_count() + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )

    print(args.rank, " gpu", args.gpu)

    torch.cuda.set_device(args.gpu)  # use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.epochs)

    #############
    # Create MONAI dataset
    training_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key="training",
        base_dir=args.data_root,
    )
    validation_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key="validation",
        base_dir=args.data_root,
    )

    if args.quick:  # for debugging on a small subset
        training_list = training_list[:16]
        validation_list = validation_list[:16]

    train_transform = Compose(
        [
            LoadImaged(keys=["image"], reader=WSIReader, backend=args.image_backend, dtype=np.uint8,
                       level=args.image_level, image_only=True),
            LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes),
            RandGridPatchd(
                keys=["image"],
                patch_size=(args.tile_size, args.tile_size),
                num_patches=args.tile_count,
                sort_fn="min",
                pad_mode=None,
                constant_values=255,
            ),
            SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=["image"], prob=0.5),
            ScaleIntensityRanged(keys=["image"], a_min=np.float32(255), a_max=np.float32(0)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    valid_transform = Compose(
        [
            LoadImaged(keys=["image"], reader=WSIReader, backend=args.image_backend, dtype=np.uint8,
                       level=args.image_level, image_only=False),
            LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes),
            GridPatchd(
                keys=["image"],
                patch_size=(args.tile_size, args.tile_size),
                threshold=0.999 * 3 * 255 * args.tile_size * args.tile_size,
                pad_mode=None,
                constant_values=255,
            ),
            SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
            ScaleIntensityRanged(keys=["image"], a_min=np.float32(255), a_max=np.float32(0)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # Preserve file names for visualization
    for dict in training_list:
        dict["image_name"] = dict["image"]
    for dict in validation_list:
        dict["image_name"] = dict["image"]

    dataset_train = Dataset(data=training_list, transform=train_transform)
    dataset_valid = Dataset(data=validation_list, transform=valid_transform)

    train_sampler = DistributedSampler(dataset_train) if args.distributed else None
    val_sampler = DistributedSampler(dataset_valid, shuffle=False) if args.distributed else None

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=False,
        multiprocessing_context="spawn" if args.workers > 0 else None,
        sampler=train_sampler,
        collate_fn=list_data_collate,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        multiprocessing_context="spawn" if args.workers > 0 else None,
        sampler=val_sampler,
        collate_fn=list_data_collate,
    )

    if args.rank == 0:
        print("Dataset training:", len(dataset_train), "validation:", len(dataset_valid))

    model = milmodel.MILModel(num_classes=args.num_classes, pretrained=True, mil_mode=args.mil_mode)

    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    if args.validate:
        # if we only want to validate existing checkpoint
        epoch_time = time.time()
        # val_loss, val_acc, qwk = val_epoch(model, valid_loader, epoch=0, args=args, max_tiles=args.tile_count)
        val_loss, val_acc, qwk, fpr, tpr, auc, precision, recall, fbeta_score = val_epoch(model, valid_loader, epoch=0,
                                                                                          args=args,
                                                                                          max_tiles=args.tile_count)
        if args.rank == 0:
            print(
                "Final validation loss: {:.4f}".format(val_loss),
                "acc: {:.4f}".format(val_acc),
                "qwk: {:.4f}".format(qwk),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

        exit(0)

    params = model.parameters()

    if args.mil_mode in ["att_trans", "att_trans_pyramid"]:
        m = model if not args.distributed else model.module
        params = [
            {"params": list(m.attention.parameters()) + list(m.myfc.parameters()) + list(m.net.parameters())},
            {"params": list(m.transformer.parameters()), "lr": 6e-6, "weight_decay": 0.1},
        ]

    optimizer = torch.optim.AdamW(params, lr=args.optim_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", writer.log_dir)
    else:
        writer = None

    # RUN TRAINING
    n_epochs = args.epochs
    val_acc_max = 0.0

    scaler = GradScaler(enabled=args.amp)

    for epoch in range(start_epoch, n_epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
            torch.distributed.barrier()

        print(args.rank, time.ctime(), "Epoch:", epoch)

        epoch_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler=scaler, epoch=epoch, args=args)

        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, n_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "acc: {:.4f}".format(train_acc),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_acc", train_acc, epoch)

        b_new_best = False
        val_acc = 0
        if (epoch + 1) % args.val_every == 0:

            epoch_time = time.time()
            # val_loss, val_acc, qwk = val_epoch(model, valid_loader, epoch=epoch, args=args, max_tiles=args.tile_count)
            val_loss, val_acc, qwk, fpr, tpr, auc, precision, recall, fbeta_score = val_epoch(model, valid_loader,
                                                                                              epoch=epoch, args=args,
                                                                                              max_tiles=args.tile_count)
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, n_epochs - 1),
                    "loss: {:.4f}".format(val_loss),
                    "acc: {:.4f}".format(val_acc),
                    "auc: {:.4f}".format(auc),
                    "precision: {:.4f}".format(precision),
                    "recall: {:.4f}".format(recall),
                    "qwk: {:.4f}".format(qwk),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )

                # send to clearml

                Logger.current_logger().report_scalar("ACC", "train_acc", iteration=epoch, value=train_acc)
                Logger.current_logger().report_scalar("ACC", "val_acc", iteration=epoch, value=val_acc)

                Logger.current_logger().report_scalar("Loss", "train_loss", iteration=epoch, value=train_loss)
                Logger.current_logger().report_scalar("Loss", "val_loss", iteration=epoch, value=val_loss)

                Logger.current_logger().report_scalar("AUC", "val_auc", iteration=epoch, value=auc)
                Logger.current_logger().report_scalar("Precision", "val_precision", iteration=epoch, value=precision)
                Logger.current_logger().report_scalar("Recall", "val_recall", iteration=epoch, value=recall)
                Logger.current_logger().report_scalar("FBeta", "val_fbeta_score", iteration=epoch, value=fbeta_score)

                if hasattr(args, 'epoch_end_callback'):
                    args.epoch_end_callback(epoch, {
                        'train_acc': train_acc.item(),
                        'val_acc': val_acc.item(),
                        'train_loss': train_loss.item(),
                        'val_loss': val_loss.item(),
                        'val_auc': auc,
                        'val_precision': precision,
                        'val_recall': recall,
                        'val_fbeta_score': fbeta_score,
                    })

                if writer is not None:
                    writer.add_scalar("val_loss", val_loss, epoch)
                    writer.add_scalar("val_acc", val_acc, epoch)
                    writer.add_scalar("val_qwk", qwk, epoch)
                    writer.add_scalar("val_auc", auc, epoch)
                    writer.add_scalar("val_precision", precision, epoch)
                    writer.add_scalar("val_recall", recall, epoch)
                    writer.add_scalar("val_fbeta_score", fbeta_score, epoch)

                val_acc = qwk

                if val_acc > val_acc_max:
                    print("qwk ({:.6f} --> {:.6f})".format(val_acc_max, val_acc))
                    val_acc_max = val_acc
                    b_new_best = True

        if args.rank == 0 and args.logdir is not None:
            save_checkpoint(model, epoch, args, best_acc=val_acc, filename="model_final.pt")
            if b_new_best:
                print("Copying to model.pt new best model!!!!")
                shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

                # send to clearml

        scheduler.step()

    print("ALL DONE")


def add_args(parser):
    parser.add_argument(
        "--data_root", default="/PandaChallenge2020/train_images/", help="path to root folder of images"
    )
    parser.add_argument("--dataset_json", default=None, type=str, help="path to dataset json file")

    parser.add_argument("--num_classes", default=5, type=int, help="number of output classes")
    parser.add_argument("--mil_mode", default="att_trans", help="MIL algorithm")
    parser.add_argument(
        "--tile_count", default=44, type=int, help="number of patches (instances) to extract from WSI image"
    )
    parser.add_argument("--tile_size", default=256, type=int, help="size of square patch (instance) in pixels")

    parser.add_argument("--checkpoint", default=None, help="load existing checkpoint")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run only inference on the validation set, must specify the checkpoint argument",
    )

    parser.add_argument("--logdir", default=None, help="path to log directory to store Tensorboard logs")

    parser.add_argument("--epochs", "--max_epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size, the number of WSI images per gpu")
    parser.add_argument("--optim_lr", default=3e-5, type=float, help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float, help="optimizer weight decay")
    parser.add_argument("--amp", action="store_true", help="use AMP, recommended")
    parser.add_argument("--val_every",
                        "--val_interval",
                        default=1,
                        type=int,
                        help="run validation after this number of epochs, default 1 to run every epoch",
                        )
    parser.add_argument("--workers", default=2, type=int, help="number of workers for data loading")

    # for multigpu
    parser.add_argument("--distributed", action="store_true", help="use multigpu training, recommended")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument(
        "--dist-url", default="tcp://127.0.0.1:23456", type=str, help="url used to set up distributed training"
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

    parser.add_argument(
        "--quick", action="store_true", help="use a small subset of data for debugging"
    )

    parser.add_argument('--project_name', type=str, default='monai-mil', help='name of project')
    parser.add_argument('--task_name', type=str, default='monai-mil_template', help='name of task')
    parser.add_argument('--image_backend', type=str, default='cucim', help='image backend to use')
    parser.add_argument('--image_level', type=int, default=1, help='image level to use')

def parse_args():
    parser = argparse.ArgumentParser(description="Multiple Instance Learning (MIL) example of classification from WSI.")
    add_args(parser)
    args = parser.parse_args()

    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")
    return args


if __name__ == "__main__":

    args = parse_args()

    task = Task.init(project_name=args.project_name, task_name=args.task_name)

    if args.dataset_json is None:
        # download default json datalist
        resource = "https://drive.google.com/uc?id=1L6PtKBlHHyUgTE4rVhRuOLTQKgD4tBRK"
        dst = "./datalist_panda_0.json"
        if not os.path.exists(dst):
            gdown.download(resource, dst, quiet=False)
        args.dataset_json = dst

    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.optim_lr = ngpus_per_node * args.optim_lr / 2  # heuristic to scale up learning rate in multigpu setup
        args.world_size = ngpus_per_node * args.world_size

        print("Multigpu", ngpus_per_node, "rescaled lr", args.optim_lr)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)
