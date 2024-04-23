import logging
import os
import sys
import tempfile
from glob import glob

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.apps import DecathlonDataset, CrossValidation
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    Compose,
    LoadImaged,
    Compose,
    LoadImaged,
    MapTransform,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandRotate90d,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd
)

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

from monai.visualize import plot_2d_or_3d_image
import monai.losses
from monai.utils import set_determinism
from monai.config import print_config
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from utils.ConvertToMultiChannelBasedOnBratsClassesd import ConvertToMultiChannelBasedOnBratsClassesd
from utils.BRATS2DSlicedDataset import BRATS2DSlicedDataset
import time
import os
from utils import utils

if __name__=="__main__":

    
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


    parser = argparse.ArgumentParser('BRATS 2D Unet', add_help=False)
    parser.add_argument('--folds', default=3, type=int, help='Number of folds')
    parser.add_argument('--val_interval', default=5, type=int, help='validation frequency')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--sanity_test', type=utils.bool_flag, default=False, help="""Do Sanity Test""")
    parser.add_argument('--val_amp', type=utils.bool_flag, default=False, help="""Use cuda.amp""")

    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=1e-4, type=float, help="""Learning rate""")

    parser.add_argument('--root_dir', default='./', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="/lustre/fs1/home/jbhol/EEG/gits/BRATS2DUnet/output", type=str, help='Path to save logs and checkpoints.')
    # parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    FLAGS = parser.parse_args()
    utils.init_distributed_mode(FLAGS)

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    root_dir = FLAGS.root_dir
    # directory = os.environ.get("MONAI_DATA_DIRECTORY")
    # root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)
    set_determinism(seed=FLAGS.seed)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=["image", "label"]),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"]),
        ]
    )


    NumberOfFolds = FLAGS.folds
    SANITY_TEST = FLAGS.sanity_test
    VAL_AMP = FLAGS.val_amp
    val_interval = FLAGS.val_interval
    BATCH_SIZE = FLAGS.batch_size
    NUMBER_OF_WORKERS = FLAGS.num_workers
    wholeTumor = True
    max_epochs = FLAGS.epochs

    print(FLAGS)
    

    cvdataset = CrossValidation(
        dataset_cls=DecathlonDataset,
        nfolds=NumberOfFolds,
        seed=FLAGS.seed,
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        transform=train_transforms,
        download=True,
        cache_rate=0.0,
    )


    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    # start a typical PyTorch training
    
    folds_numbers = [i for i in range(NumberOfFolds)]
    for fold in folds_numbers:
        total_start = time.time()
        
        current_folds_to_train = [i for i in range(NumberOfFolds)]
        current_folds_to_train.pop(current_folds_to_train.index(fold))
        print("Folds ",current_folds_to_train, "will be trained and ", fold, " will be used for validation")

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        hd_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")
        hd_metric_batch = HausdorffDistanceMetric(include_background=True, reduction="mean_batch")

        train_ds = cvdataset.get_dataset(folds=current_folds_to_train, download=True)
        val_ds = cvdataset.get_dataset(folds=fold, transform=val_transforms, download=True)
        print("Train ds ",len(train_ds), " val ds ", len(val_ds))

        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
        
        # loss_function = monai.losses.DiceLoss(sigmoid=True)
        loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
        # optimizer = torch.optim.Adam(model.parameters(), 1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        # use amp to accelerate training
        scaler = torch.cuda.amp.GradScaler()
        # enable cuDNN benchmark
        torch.backends.cudnn.benchmark = True

        best_metric = -1
        best_metric_epoch = -1
        best_hd_metric_value= -1
        epoch_loss_values = list()
        metric_values = list()
        # writer = SummaryWriter()

        best_metrics_epochs_and_time = [[], [], []]
        best_hd_metrics_epochs_and_time = [[], [], []]

        epoch_loss_values = []
        metric_values = []
        metric_values_tc = [] # tumor core (TC)
        metric_values_wt = [] # whole tumor
        metric_values_et = [] # and enhanced tumor (ET) 


        hd_metric_values = []
        hd_metric_values_tc = [] # tumor core (TC)
        hd_metric_values_wt = [] # whole tumor
        hd_metric_values_et = [] # and enhanced tumor (ET) 
        
        

        for epoch in range(max_epochs):
            epoch_start = time.time()
            model.train()
            epoch_loss = 0
            step = 0
            TotalData = 0
            TempData = []
            # print("Appending Slices")
            local_batch_size = BATCH_SIZE
            TotalBatch = len(train_ds)//local_batch_size
            
            ds_idx_total = -1
            total_batch = -1
            
            for ds_idx, data in enumerate(train_ds):
                TempData.append(data)
                
                # print(f"added : {ds_idx}")
                if len(TempData)>local_batch_size or ds_idx==len(train_ds):
                    Temp2dDataset = BRATS2DSlicedDataset(dataset_slices=TempData, channels_to_use=0, wholeTumor=True)
                    TotalData += len(Temp2dDataset)
                    # print(f"Temp2dDataset : {len(Temp2dDataset)} Total {TotalData}")
                    TempData = []
                    Temp2dDataset_Loader = DataLoader(Temp2dDataset, batch_size=BATCH_SIZE, num_workers=NUMBER_OF_WORKERS, pin_memory=torch.cuda.is_available())
                    
                    total_batch += int(len(Temp2dDataset)/BATCH_SIZE)
                    
                    
                    for b_idx, batch_data in enumerate(Temp2dDataset_Loader):
                        ds_idx_total +=1
                        step += 1
                        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = loss_function(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        # optimizer.zero_grad()
                        # with torch.cuda.amp.autocast():
                        #     outputs = model(inputs)
                        #     loss = loss_function(outputs, labels)
          
                        # scaler.scale(loss).backward()
                        # scaler.step(optimizer)
                        # scaler.update()
                        # epoch_loss += loss.item()
                        # print(f"Batch loss:  {loss.item()} ")
                        
                        
                        print(f"[{ds_idx_total}/{total_batch}] loss: {loss.item()}")
          
                    del Temp2dDataset, Temp2dDataset_Loader
                    if SANITY_TEST and ds_idx>10:break
                    
            
            lr_scheduler.step()
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_images = None
                    val_labels = None
                    val_outputs = None
                    TempData = []
                    TotalData = 0
                    val_idx_total = 0
                    total_val_batch = 0
                    for val_ds_idx, val_ds_data in enumerate(val_ds):
                        TempData.append(val_ds_data)
                        
                        
                        if len(TempData)>local_batch_size or val_ds_idx==len(val_ds):
                            Temp2dDataset = BRATS2DSlicedDataset(dataset_slices=TempData, channels_to_use=0, wholeTumor=True)
                            TotalData += len(Temp2dDataset)
                            Temp2dDataset_Loader = DataLoader(Temp2dDataset, batch_size=BATCH_SIZE, num_workers=NUMBER_OF_WORKERS, pin_memory=torch.cuda.is_available())
                            TempData  = []
                            
                            total_val_batch += int(len(Temp2dDataset)/BATCH_SIZE)
                            
                            for val_idx, val_data in enumerate(Temp2dDataset_Loader):
                                val_idx_total +=1
                                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                                roi_size = (96, 96)
                                sw_batch_size = 4
        
                                if VAL_AMP:
                                    with torch.cuda.amp.autocast():
                                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                                else:
                                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                                
                                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                                # compute metric for current iteration
                                dice_metric(y_pred=val_outputs, y=val_labels)
                                hd_metric(y_pred=val_outputs, y=val_labels)
                                
                                print(f"[{val_idx_total}/{total_val_batch}] Validated")
        
                            if SANITY_TEST and val_ds_idx>10:break
                            del Temp2dDataset, Temp2dDataset_Loader
                            
                    metric = dice_metric.aggregate().item()
                    metric_values.append(metric)
                    dice_metric.reset()

                    hd_metric_value = hd_metric.aggregate().item()
                    hd_metric_values.append(hd_metric_value)
                    hd_metric.reset()

                    if hd_metric_value > best_hd_metric_value:
                        best_hd_metric_value = hd_metric_value
                        best_hd_metric_epoch = epoch + 1
                        best_hd_metrics_epochs_and_time[0].append(best_hd_metric_value)
                        best_hd_metrics_epochs_and_time[1].append(best_hd_metric_epoch)
                        best_hd_metrics_epochs_and_time[2].append(time.time() - total_start)

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        best_metrics_epochs_and_time[0].append(best_metric)
                        best_metrics_epochs_and_time[1].append(best_metric_epoch)
                        best_metrics_epochs_and_time[2].append(time.time() - total_start)
                        torch.save(
                            model.state_dict(),
                            os.path.join(root_dir, f"{FLAGS.output_dir}/fold_{fold}_best_metric_model.pth"),
                        )
                        print("saved new best metric model")
                    
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f" hd: {hd_metric_value:.4f} "
                        f"\nbest mean dice: {best_metric:.4f} best hd: {best_hd_metric_value:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
                    # writer.add_scalar("val_mean_dice", metric, epoch + 1)
                        # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                        # plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                        # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                        # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")



            epoch_end = time.time()
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}  time: {(epoch_end-epoch_start)/60} minutes")
            
            if SANITY_TEST and epoch>5:
                break

        
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y, color="red")
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel("epoch")
        plt.plot(x, y, color="green")
        plt.savefig(os.path.join(f"{FLAGS.output_dir}/fold_{fold}_best_metric_model_Epoch_Average_Loss.png"))
        # plt.show()