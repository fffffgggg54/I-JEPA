import torch
import torch.cuda.amp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.profiler
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DistributedSampler
import torchvision.datasets as dset
from torchvision import models, transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd
import numpy as np
import time
import glob
import gc
import os
import torchvision
import torch_optimizer

import multiprocessing

import timm
import timm.optim
import transformers

import timm.layers.ml_decoder as ml_decoder
import timm.layers
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, AsymmetricLossSingleLabel
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.mixup import FastCollateMixup, Mixup

from PIL import Image, ImageOps, ImageDraw
import PIL
import random

from i_jepa import I_JEPA


torch.backends.cudnn.benchmark = True

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
#torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
#torch.backends.cudnn.allow_tf32 = True

timm.layers.fast_norm.set_fast_norm(enable=True)

# ================================================
#           CONFIGURATION OPTIONS
# ================================================

# TODO use a configuration file or command line arguments instead of having a bunch of variables

FLAGS = {}

# path config for various directories and files
# TODO replace string appending with os.path.join()

FLAGS['rootPath'] = "/media/fredo/SAMSUNG_500GB/imagenet/"
FLAGS['imageRoot'] = FLAGS['rootPath'] + ''

#FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/IJEPA_base_patch16_224/'
FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/IJEPA_regnety_800/'


# device config


FLAGS['ngpu'] = torch.cuda.is_available()
FLAGS['device'] = torch.device("cuda:0" if (torch.cuda.is_available() and FLAGS['ngpu'] > 0) else "mps" if (torch.has_mps == True) else "cpu")
FLAGS['device2'] = FLAGS['device']
if(torch.has_mps == True): FLAGS['device2'] = "cpu"
FLAGS['use_AMP'] = False
FLAGS['use_ddp'] = True
#FLAGS['use_scaler'] = FLAGS['use_AMP']
FLAGS['use_scaler'] = False
#if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

# dataloader config

FLAGS['num_workers'] = 10


# training config

FLAGS['num_epochs'] = 100
FLAGS['batch_size'] = 128
FLAGS['gradient_accumulation_iterations'] = 1

FLAGS['base_learning_rate'] = 1e-3
FLAGS['base_batch_size'] = 1024
FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
FLAGS['lr_warmup_epochs'] = 5

FLAGS['weight_decay'] = 1e-2

FLAGS['resume_epoch'] = 0

FLAGS['finetune'] = False

FLAGS['image_size'] = 224
FLAGS['progressiveImageSize'] = False
FLAGS['progressiveSizeStart'] = 0.5
FLAGS['progressiveAugRatio'] = 3.0

FLAGS['crop'] = 0.875
FLAGS['interpolation'] = torchvision.transforms.InterpolationMode.BICUBIC
FLAGS['image_size_initial'] = int(round(FLAGS['image_size'] // FLAGS['crop']))

FLAGS['channels_last'] = False
FLAGS['compile_model'] = False

# debugging config

FLAGS['verbose_debug'] = False
FLAGS['skip_test_set'] = False
FLAGS['stepsPerPrintout'] = 50
FLAGS['val'] = False

classes = None


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir

def getData():
    startTime = time.time()

    #trainSet = torchvision.datasets.ImageNet(FLAGS['imageRoot'], split = 'train')
    #testSet = torchvision.datasets.ImageNet(FLAGS['imageRoot'], split = 'val')
    
    trainSet = timm.data.create_dataset('', root=FLAGS['imageRoot'], split = 'train', batch_size=FLAGS['batch_size'])
    testSet = timm.data.create_dataset('', root=FLAGS['imageRoot'], split = 'validation', batch_size=FLAGS['batch_size'])

    global classes
    #classes = {classIndex : className for classIndex, className in enumerate(trainSet.classes)}
    
    classes = trainSet.reader.class_to_idx
    
    image_datasets = {'train': trainSet, 'val' : testSet}   # put dataset into a list for easy handling
    return image_datasets

def modelSetup(classes):
    
    '''
    myCvtConfig = transformers.CvtConfig(num_channels=3,
        patch_sizes=[7, 5, 3, 3],
        patch_stride=[4, 3, 2, 2],
        patch_padding=[2, 2, 1, 1],
        embed_dim=[64, 240, 384, 896],
        num_heads=[1, 3, 6, 14],
        depth=[1, 4, 8, 16],
        mlp_ratio=[4.0, 4.0, 4.0, 4.0],
        attention_drop_rate=[0.0, 0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.0, 0.1],
        qkv_bias=[True, True, True, True],
        cls_token=[False, False, False, True],
        qkv_projection_method=['dw_bn', 'dw_bn', 'dw_bn', 'dw_bn'],
        kernel_qkv=[3, 3, 3, 3],
        padding_kv=[1, 1, 1, 1],
        stride_kv=[2, 2, 2, 2],
        padding_q=[1, 1, 1, 1],
        stride_q=[1, 1, 1, 1],
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        num_labels=len(classes))
    '''
    
    # custom cvt
    
    #model = transformers.CvtForImageClassification(myCvtConfig)
    
    # pytorch builtin models
    
    #model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    #model = models.resnet152()
    #model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    #model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    #model = models.resnet34()
    #model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
    #model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    
    #model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    # regular timm models
    
    #model = timm.create_model('maxvit_tiny_tf_224.in1k', pretrained=True, num_classes=len(classes))
    #model = timm.create_model('ghostnet_050', pretrained=True, num_classes=len(classes))
    model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=0, global_pool='', drop_path_rate=0.1)
    #model = timm.create_model('vit_small_resnet26d_224', pretrained=False, num_classes=len(classes), drop_rate = 0., drop_path_rate = 0.1)
    
    #model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0, drop_rate = 0.0, drop_path_rate = 0.2, global_pool='', class_token=False)
    model = I_JEPA(model)
    
    
    

    
    #model = ml_decoder.add_ml_decoder_head(model)
    
    # cvt
    
    #model = transformers.CvtForImageClassification.from_pretrained('microsoft/cvt-13')
    #model.classifier = nn.Linear(model.config.embed_dim[-1], len(classes))

    # regular huggingface models

    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/levit-384", num_labels=len(classes), ignore_mismatched_sizes=True)
    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", num_labels=len(classes), ignore_mismatched_sizes=True)
    
    
    # modified timm models with custom head with hidden layers
    '''
    model = timm.create_model('mixnet_s', pretrained=True, num_classes=-1) # -1 classes for identity head by default
    
    model = nn.Sequential(model,
                          nn.LazyLinear(len(classes)),
                          nn.ReLU(),
                          nn.Linear(len(classes), len(classes)))
    
    '''
    
    
        
    
    #model.reset_classifier(len(classes))
    
    
    return model

def getDataLoader(dataset, batch_size, epoch):
    distSampler = DistributedSampler(dataset=dataset, seed=17, drop_last=True)
    distSampler.set_epoch(epoch)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler=distSampler, num_workers=FLAGS['num_workers'], persistent_workers = True, prefetch_factor=2, pin_memory = True, generator=torch.Generator().manual_seed(41))

def trainCycle(image_datasets, model):

    is_head_proc = dist.get_rank() == 0


    if(is_head_proc): print("starting training")
    startTime = time.time()

    
    #dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=FLAGS['batch_size'], shuffle=True, num_workers=FLAGS['num_workers'], persistent_workers = False, prefetch_factor=2, pin_memory = True, drop_last=True, generator=torch.Generator().manual_seed(41)) for x in image_datasets} # set up dataloaders
    
    
    #mixup = Mixup(mixup_alpha = 0.2, cutmix_alpha = 0, label_smoothing=0.1)
    #dataloaders['train'].collate_fn = mixup_collate
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    device = FLAGS['device']
    device2 = FLAGS['device2']
    
    #memory_format = torch.channels_last
    memory_format = torch.channels_last if FLAGS['channels_last'] else torch.contiguous_format
    
    
    model = model.to(device, memory_format=memory_format)
    
    
    
    if (FLAGS['resume_epoch'] > 0) and is_head_proc:
        state_dict = torch.load(FLAGS['modelDir'] + 'saved_model_epoch_' + str(FLAGS['resume_epoch'] - 1) + '.pth', map_location=torch.device('cpu'))
        #out_dict={}
        #for k, v in state_dict.items():
        #    k = k.replace('_orig_mod.', '')
        #    k = k.replace('module.', '')
        #    out_dict[k] = v
            
        model.load_state_dict(state_dict)
        


    if (FLAGS['use_ddp'] == True):
        
        model = DDP(model, device_ids=[FLAGS['device']], gradient_as_bucket_view=True)
        
    if(FLAGS['compile_model'] == True):
        model = torch.compile(model)
        
    
    # initialize jepa params
    with torch.no_grad():
        in_tensor = torch.randn(FLAGS['batch_size'], 3, FLAGS['image_size'], FLAGS['image_size'], device=device)
        out_tensor = model(in_tensor)
        del in_tensor
        del out_tensor
        torch.cuda.empty_cache()
        
    if(is_head_proc): print(torch.cuda.mem_get_info())
    
    if(is_head_proc): print("initialized training, time spent: " + str(time.time() - startTime))


    #criterion = SoftTargetCrossEntropy()
    criterion = nn.MSELoss(reduction='sum')
    # CE with ASL (both gammas 0), eps controls label smoothing, pref sum reduction
    #criterion = AsymmetricLossSingleLabel(gamma_pos=0, gamma_neg=0, eps=0.0, reduction = 'mean')
    #criterion = nn.BCEWithLogitsLoss()

    #optimizer = optim.Adam(params=parameters, lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.SGD(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'], momentum = 0.9)
    #optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    optimizer = timm.optim.Adan(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    if (FLAGS['resume_epoch'] > 0):
        optimizer.load_state_dict(torch.load(FLAGS['modelDir'] + 'optimizer' + '.pth'))
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])
    #scheduler.last_epoch = len(dataloaders['train'])*FLAGS['resume_epoch']
    
    if (FLAGS['use_scaler'] == True): scaler = torch.cuda.amp.GradScaler()

    
    if(is_head_proc): print("starting training")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(FLAGS['resume_epoch'], FLAGS['num_epochs']):
        epochTime = time.time()
        if(is_head_proc): print("starting epoch: " + str(epoch))
        
        
        dataloaders = {x: getDataLoader(image_datasets[x], FLAGS['batch_size'], epoch) for x in image_datasets} # set up dataloaders

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])
        scheduler.last_epoch = len(dataloaders['train'])*epoch

        
        if(is_head_proc):
            print("starting epoch: " + str(epoch))
        
        if FLAGS['progressiveImageSize'] == True:
                    
                    
            dynamicResizeDim = int(FLAGS['image_size'] * FLAGS['progressiveSizeStart'] + epoch * \
                (FLAGS['image_size']-FLAGS['image_size'] * FLAGS['progressiveSizeStart'])/FLAGS['num_epochs'])
        else:
            dynamicResizeDim = FLAGS['image_size']
        
        
        print(f'Using image size of {dynamicResizeDim}x{dynamicResizeDim}')
        
        trainTransforms = transforms.Compose([
            transforms.Resize((dynamicResizeDim, dynamicResizeDim), interpolation = FLAGS['interpolation']),
            #torchvision.transforms.RandomResizedCrop((dynamicResizeDim, dynamicResizeDim), interpolation = FLAGS['interpolation']),
            transforms.RandomHorizontalFlip(),
            #transforms.TrivialAugmentWide(),
            #transforms.RandAugment(magnitude = epoch, num_magnitude_bins = int(FLAGS['num_epochs'] * FLAGS['progressiveAugRatio'])),
            #transforms.RandAugment(),
            #CutoutPIL(cutout_factor=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

        
        '''
        trainTransformsRSB = transforms.Compose([transforms.Resize((256)),
            transforms.RandomHorizontalFlip(),
            RandomResizedCropAndInterpolation(size=224),
            rand_augment_transform(
                config_str='rand-m9-mstd0.5', 
                hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
            ),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            RandomErasing(probability=0.5, mode='pixel', device='cpu'),
            #transforms.GaussianBlur(kernel_size=(7, 7), sigma=(2, 10)),
            #transforms.ToPILImage(),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        '''
        
        
        image_datasets['train'].transform = trainTransforms
        
        image_datasets['val'].transform = transforms.Compose([
            transforms.Resize(FLAGS['image_size_initial'], interpolation = FLAGS['interpolation']),
            transforms.CenterCrop((int(FLAGS['image_size']),int(FLAGS['image_size']))),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        
        for phase in ['train', 'val']:
        
            samples = 0
            correct = 0
            
            if phase == 'train':
                model.train()  # Set model to training mode
                print("training set")
                
                
            if phase == 'val' and FLAGS['val'] == False and is_head_proc:
                modelDir = create_dir(FLAGS['modelDir'])
                state_dict = model.state_dict()
                    
                out_dict={}
                for k, v in state_dict.items():
                    k = k.replace('_orig_mod.', '')
                    k = k.replace('module.', '')
                    out_dict[k] = v
                
                torch.save(out_dict, modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')

                torch.save(optimizer.state_dict(), modelDir + 'optimizer' + '.pth')
                model.eval()   # Set model to evaluate mode
                print("validation set")
                if(FLAGS['skip_test_set'] == True):
                    print("skipping...")
                    break;

            loaderIterable = enumerate(dataloaders[phase])
            for i, (images, tags) in loaderIterable:
                #if phase == 'train':
                #    images, tags = mixup(images, tags)
                

                imageBatch = images.to(device, memory_format=memory_format, non_blocking=True)
                tagBatch = tags.to(device, non_blocking=True)
                
                if(is_head_proc): print(torch.cuda.mem_get_info())
                
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=FLAGS['use_AMP']):
                        
                        
                        
                        outputs = model(imageBatch)
                        if(is_head_proc): print(torch.cuda.mem_get_info())
                        #outputs = model(imageBatch).logits
                        #if phase == 'val':
                        '''
                        with torch.no_grad():
                            #preds = torch.argmax(outputs, dim=1)
                            preds = torch.softmax(outputs, dim=1) if phase == 'train' else torch.argmax(outputs, dim=1)
                            
                            
                            samples += len(images)
                            #correct += sum(preds == tagBatch)
                            correct += (preds * tagBatch).sum() if phase == 'train' else sum(preds == tagBatch)
                        '''    
                        #print(tagBatch.shape)
                        #if phase == 'val':
                        #    tagBatch=torch.zeros([FLAGS['batch_size'], len(classes)]).scatter_(1, tags.view(FLAGS['batch_size'], 1), 1)
                        #loss = criterion(outputs.to(device2), tagBatch.to(device2))
                        
                        loss = criterion(outputs[0], outputs[1])
                        

                        # backward + optimize only if in training phase
                        if phase == 'train' and (loss.isnan() == False):
                            if (FLAGS['use_scaler'] == True):   # cuda gpu case
                                with model.no_sync():
                                    scaler.scale(loss).backward()
                                if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad(set_to_none=True)
                            else:                               # apple gpu/cpu case
                                with model.no_sync():
                                    loss.backward()
                                if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                                    optimizer.step()
                                    optimizer.zero_grad(set_to_none=True)
                        
                        
                            torch.cuda.synchronize()
                                    
                if i % stepsPerPrintout == 0:
                    #accuracy = 100 * (correct/(samples+1e-8))
                    
                    #print(outputs[0])
                    #print(outputs[1])
                    

                    imagesPerSecond = (dataloaders[phase].batch_size*stepsPerPrintout)/(time.time() - cycleTime)
                    cycleTime = time.time()
                    
                    if(FLAGS['use_ddp'] == True):
                        imagesPerSecond = torch.Tensor([imagesPerSecond]).to(device)
                        torch.distributed.all_reduce(imagesPerSecond, op = torch.distributed.ReduceOp.SUM)
                        imagesPerSecond = imagesPerSecond.cpu()
                        imagesPerSecond = imagesPerSecond.item()

                    if(is_head_proc): print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f' % (epoch, FLAGS['num_epochs'], i, len(dataloaders[phase]), loss, imagesPerSecond))
                    torch.cuda.empty_cache()

                if phase == 'train':
                    scheduler.step()
            #if phase == 'val':
            #    print(f'top-1: {100 * (correct/samples)}%')
        
        time_elapsed = time.time() - epochTime
        if(is_head_proc): print(f'epoch {epoch} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        #print(best)
        

        gc.collect()

        if(is_head_proc): print()



def main():
    #gc.set_debug(gc.DEBUG_LEAK)
    # load json files
    
    if FLAGS['use_ddp']:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        FLAGS['device'] = rank % torch.cuda.device_count()
        torch.cuda.set_device(FLAGS['device'])
        torch.cuda.empty_cache()
        FLAGS['learning_rate'] *= dist.get_world_size()
    
    image_datasets = getData()
    model = modelSetup(classes)
    trainCycle(image_datasets, model)


if __name__ == '__main__':
    main()