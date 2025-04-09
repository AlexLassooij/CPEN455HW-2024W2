
import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-w', '--en_wandb', type=bool, default=False,
                            help='Enable wandb logging')
    parser.add_argument('-t', '--tag', type=str, default='default',
                            help='Tag for this run')
    
    # sampling
    parser.add_argument('-c', '--sampling_interval', type=int, default=5,
                        help='sampling interval')
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    # add a sub folder for generated samples
    parser.add_argument('-sd', '--sample_dir',  type=str, default='samples',
                        help='Location for saving samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='cpen455', help='Can be either cifar|mnist|cpen455')
    parser.add_argument('-st', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('--obs', type=tuple, default=(3, 32, 32),
                        help='Observation shape')
    
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=1,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=40,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-sb', '--sample_batch_size', type=int, default=32,
                        help='Batch size during sampling per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=5000, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    parser.add_argument('sc', '--sample_class', type=int, default=0,
                    help='Generate samples for a specific class (0-based index)')
    print('......sampling......')

    args = parser.parse_args()
    pprint(args.__dict__)
    check_dir_and_create(args.save_dir)
    model_name = 'pcnn_' + args.dataset + "_"
    model_path = args.save_dir + '/'

    # make sure this is enabled when generating
    if args.load_params is not None:
        model_name = model_name + 'load_model'
        model_path = model_path + model_name + '/'
    else:
        model_name = model_name + 'from_scratch'
        model_path = model_path + model_name + '/'

    input_channels = args.obs[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalPixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix,
            num_classes=4, embedding_dim=32)
    model = model.to(device)
    
    if args.load_params:
        model.load_state_dict(torch.load(args.load_params))
        print('model parameters loaded')

    job_name = "PCNN_Generating_" + "class :" + args.sample_class

    if args.en_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set entity to specify your username or team name
            # entity="qihangz-work",
            # set the wandb project where this run will be logged
            project="CPEN455HW",
            # group=Group Name
            name=job_name,
        )
        wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        wandb.config.update(args)

    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    sample_t = sample_conditional(model, args.sample_batch_size, args.obs, sample_op, args.sample_class)
    sample_t = rescaling_inv(sample_t)
    save_images(sample_t, args.sample_dir)
    sample_result = wandb.Image(sample_t, caption="class {}".format(args.sample_class))

    gen_data_dir = args.sample_dir
    ref_data_dir = args.data_dir +'/test'
    paths = [gen_data_dir, ref_data_dir]
    try:
        fid_score = calculate_fid_given_paths(paths, 32, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score))
    except:
        print("Dimension {:d} fails!".format(192))
        
    if args.en_wandb:
        wandb.log({"samples": sample_result,
                    "FID": fid_score})