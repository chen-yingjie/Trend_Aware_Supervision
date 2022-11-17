import numpy as np
import os
import argparse
from ruamel import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--processor_name',
                        type=str,
                        default='train-TAS',
                        help='processor name')
    parser.add_argument('-c',
                        '--config_dir',
                        type=str,
                        default='./config/exp',
                        help='config dir name')
    parser.add_argument('-w',
                        '--work_dir',
                        type=str,
                        default='./work_dir/train/bp4d/exp',
                        help='work dir name')
    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        default='./data/bp4d_example',
                        help='data dir name')
    parser.add_argument('--num_class',
                        type=int,
                        default=5,
                        help='num of class to detect')

    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        os.mkdir(args.config_dir)

    for k in [1]:

        desired_caps = {
            'work_dir': os.path.join(args.work_dir, str(k)),
            'feeder': 'feeder.feeder_segment.Feeder',
            'train_feeder_args': {
                'label_path':
                os.path.join(args.data_dir, 'train' + str(k) + '_label.pkl'),
                'trend_path':
                os.path.join(args.data_dir, 'train' + str(k) + '_trend.pkl'),
                'state_path':
                os.path.join(args.data_dir, 'train' + str(k) + '_state.pkl'),
                'image_path':
                os.path.join(args.data_dir,
                             'train' + str(k) + '_imagepath.pkl'),
                'image_size':
                256,
                'frames_between_keyframe':
                16,
                'sampling_strategy':
                'linear',
                'smooth': False,
                'istrain':
                True,
                'isaug': False,
            },
            'test_feeder_args': {
                'label_path':
                os.path.join(args.data_dir, 'val' + str(k) + '_label.pkl'),
                'trend_path':
                os.path.join(args.data_dir, 'val' + str(k) + '_trend.pkl'),
                'state_path':
                os.path.join(args.data_dir, 'val' + str(k) + '_state.pkl'),
                'image_path':
                os.path.join(args.data_dir, 'val' + str(k) + '_imagepath.pkl'),
                'image_size':
                256,
                'istrain':
                False
            },
            'batch_size': 16,
            'test_batch_size': 16,
            'num_worker': 1,
            'debug': False,
            'model': 'net.TAS.Model',
            'model_args': {
                'num_class': args.num_class,
                'backbone': 'resnet34'
            },
            'log_interval': 500,
            'save_interval': 15,
            'device': [0],
            'base_lr': 0.005,
            'lr_backbone': 0.001,
            'scheduler': 'constant',
            'scheduler_args': {
                'lr_decay_rate': 0.3,
                'lr_decay_epochs': [],
            },
            'num_epoch': 20,
            'optimizer': 'SGD',
            'weight_decay': 0.0005,
            'pretrain': True,
            'warmup_epoch': -1,
            'loss': ['regression', 'monotonic_trend', 'vicinal_distribution', 'subject_variance'],
            'loss_args': {
                'regression': {
                    'loss_lambda': 1,
                },
                'monotonic_trend': {
                    'loss_lambda': 0.1,
                },
                'vicinal_distribution': {
                    'loss_lambda': 0.05,
                },
                'subject_variance': {
                    'num_class': args.num_class,
                    'loss_lambda': 0.005,
                },
            },
            'mixup_seg': True,
            'alpha': 0.5,
        }

        yamlpath = os.path.join(args.config_dir, 'train' + str(k) + '.yaml')
        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(desired_caps, f, Dumper=yaml.RoundTripDumper)

        cmdline = "python main.py " + args.processor_name + " -c " + yamlpath
        print(cmdline)
        os.system(cmdline)
