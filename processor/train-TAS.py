#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import os
import math

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction

from .processor import Processor
from tools.utils import funcs, losses

from tensorboardX import SummaryWriter


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def init_environment(self):

        super().init_environment()

    def load_model(self):

        self.train_logger = SummaryWriter(log_dir=os.path.join(
            self.arg.work_dir, 'train'),
                                          comment='train')
        self.validation_logger = SummaryWriter(log_dir=os.path.join(
            self.arg.work_dir, 'validation'),
                                               comment='validation')

        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))

        update_dict = {}
        model_dict = self.model.state_dict()
        if self.arg.pretrain and self.arg.model_args['backbone'] == 'resnet34':
            pretrained_dict = models.resnet34(pretrained=True).state_dict()
            for k, v in pretrained_dict.items():
                if "layer1" in k:
                    update_dict[k.replace("layer1", "encoder.4", 1)] = v
                elif "layer2" in k:
                    update_dict[k.replace("layer2", "encoder.5", 1)] = v
                elif "layer3" in k:
                    update_dict[k.replace("layer3", "encoder.6", 1)] = v
                elif "layer4" in k and self.arg.model_args[
                        'backbone'] == 'resnet34':
                    update_dict[k.replace("layer4", "encoder.7", 1)] = v
                elif k == 'conv1.weight':
                    update_dict['encoder.0.weight'] = v
                elif k == 'bn1.weight':
                    update_dict['encoder.1.weight'] = v
                elif k == 'bn1.bias':
                    update_dict['encoder.1.bias'] = v
                elif k == 'bn1.running_mean':
                    update_dict['encoder.1.running_mean'] = v
                elif k == 'bn1.running_var':
                    update_dict['encoder.1.running_var'] = v
                elif k == 'bn1.num_batches_tracked':
                    update_dict['encoder.1.num_batches_tracked'] = v
        elif not self.arg.pretrain and self.arg.resume and os.path.isfile(
                self.arg.resume):
            print("Loading checkpoint '{}'".format(self.arg.resume))
            checkpoint = torch.load(self.arg.resume)
            update_dict = checkpoint

        print('updated params:{}'.format(len(update_dict)))
        model_dict.update(update_dict)
        self.model.load_state_dict(model_dict)

        if isinstance(self.arg.loss, list):
            self.loss = dict()
            if 'regression' in self.arg.loss:
                self.loss['regression'] = losses.RegressionLoss(
                    **self.arg.loss_args['regression'])
            if 'monotonic_trend' in self.arg.loss:
                self.loss['monotonic_trend'] = losses.MonotonicTrendLoss(
                    **self.arg.loss_args['monotonic_trend'])
            if 'vicinal_distribution' in self.arg.loss:
                self.loss['vicinal_distribution'] = losses.VicinalDistributionLoss(
                    **self.arg.loss_args['vicinal_distribution'])
            if 'subject_variance' in self.arg.loss:
                self.loss['subject_variance'] = losses.SubjectVarianceLoss(
                    **self.arg.loss_args['subject_variance'])
        else:
            raise ValueError()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       nesterov=self.arg.nesterov,
                                       weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                           lr=self.arg.base_lr,
                                           alpha=0.9,
                                           momentum=0,
                                           weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.scheduler == 'cosine':
            eta_min = self.arg.base_lr * (
                self.arg.scheduler_args['lr_decay_rate']**3)
            self.lr = eta_min + (self.arg.base_lr - eta_min) * (1 + math.cos(
                math.pi * self.meta_info['epoch'] / self.arg.num_epoch)) / 2
        elif self.arg.scheduler == 'step':
            steps = np.sum(self.meta_info['epoch'] > np.array(
                self.arg.scheduler_args['lr_decay_epochs']))
            if steps > 0:
                self.lr = self.arg.base_lr * (
                    self.arg.scheduler_args['lr_decay_rate']**steps)
        elif self.arg.scheduler == 'constant':
            self.lr = self.arg.base_lr
        else:
            raise ValueError('Invalid learning rate schedule {}'.format(
                self.args.scheduler))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        return self.lr

    def mixup_data(self, x, y, alpha=1.0, lam=None, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if lam is None:
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam, state=None):
        return lam * criterion(pred, y_a, state=state) + (1 - lam) * criterion(
            pred, y_b, state=state)

    def train(self):

        self.model.train()
        self.adjust_lr()
        
        loader = self.data_loader['train']
        loss_value = dict()
        loss_dict = dict()
        result_frag = dict()
        label_frag = dict()
        for au in range(self.arg.model_args['num_class']):
            label_frag[au] = list()
            result_frag[au] = list()

        print('training dataloder length: ', len(loader))

        for image, label, state, au_class in loader:

            # get data
            image = image.float().to(self.dev)
            label = label.float().to(self.dev)
            state = state.float().to(self.dev)
            au_class = au_class.to(self.dev)

            # forward
            output, feature, _ = self.model(image)

            # mixup for consistency
            if self.arg.mixup_seg:
                output_mix = []
                feature_mix = []
                label_mix = []
                batch_size = image.shape[0]
                output_reshape = output.view(batch_size, -1, output.shape[-1])
                for bz in range(batch_size):
                    image_before_mix = image[bz, :, :, :, :]
                    output_before_mix = output_reshape[bz, :, :]
                    image_mix, label_a, label_b, lam = self.mixup_data(
                        image_before_mix, output_before_mix, self.arg.alpha)

                    output_mix_tmp, feature_mix_tmp, _ = self.model(image_mix)
                    label_mix_tmp = (lam * label_a +
                                    (1 - lam) * label_b).detach()

                    output_mix.append(output_mix_tmp)
                    feature_mix.append(feature_mix_tmp)
                    label_mix.append(label_mix_tmp)

                output_mix = torch.cat(output_mix)
                feature_mix = torch.cat(feature_mix)
                label_mix = torch.cat(label_mix)

            loss = torch.tensor(0).float().to(self.dev)
            au_class_ = au_class.repeat(label.shape[1],
                                       1).permute(1, 0).reshape(-1, 1)
            output_ = torch.gather(output, 1, au_class_).reshape(-1, 1)
            label_ = torch.gather(label.reshape(-1, label.shape[-1]), 1,
                                  au_class_)

            if len(feature.shape) > 2:
                feature_ = torch.gather(
                    feature, 2,
                    au_class_.repeat(
                        1, feature.shape[1]).unsqueeze(-1)).squeeze(-1)

            for k, v in self.loss.items():
                if k in ['monotonic_trend']:
                    loss_dict[k] = self.loss[k](feature_,
                                                label_,
                                                au_class=au_class_,
                                                num_seg=image.shape[0],
                                                state=state)
                elif k in ['vicinal_distribution']:
                    loss_dict[k] = 0
                    if self.arg.mixup_seg:
                        loss_dict[k] += self.loss[k](output_mix, label_mix)
                elif k in ['subject_variance']:
                    loss_dict[k] = self.loss[k](feature,
                                                output,
                                                au_class=au_class_,
                                                num_seg=image.shape[0],
                                                epoch=self.meta_info['epoch'],
                                                state=state)
                else:
                    loss_dict[k] = self.loss[k](output_,
                                                label_,
                                                au_class=au_class_,
                                                num_seg=image.shape[0],
                                                state=state)
                
                loss += loss_dict[k]

            # log
            output_ = output_.data.cpu().numpy()
            label_ = label_.data.cpu().numpy()
            au_class_ = au_class_.data.cpu().numpy()
            for au in range(self.arg.model_args['num_class']):
                result_frag[au].append(output_[au_class_[:, ] == au])
                label_frag[au].append(label_[au_class_[:, ] == au])

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()

            if 'total' not in loss_value:
                loss_value['total'] = []
            loss_value['total'].append(self.iter_info['loss'])
            for k, v in self.loss.items():
                self.iter_info[k] = loss_dict[k].data.item()
                if k not in loss_value:
                    loss_value[k] = []
                loss_value[k].append(self.iter_info[k])

            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value['total'])
        for k, v in self.loss.items():
            self.epoch_info['mean_' + k] = np.mean(loss_value[k])

        self.show_epoch_info()
        self.io.print_timer()

        # visualize loss and metrics
        icc = []
        mae = []
        for au in range(self.arg.model_args['num_class']):
            result = np.hstack(result_frag[au]).reshape(-1, 1)
            label = np.hstack(label_frag[au]).reshape(-1, 1)
            icc_, mae_ = funcs.record_intensity_metrics(
                result, label, self.epoch_info['mean_loss'],
                self.arg.model_args['num_class'], self.arg.work_dir, 'seg')
            icc.append(icc_)
            mae.append(mae_)

        res_txt_path = os.path.join(self.arg.work_dir, 'log.txt')
        fp = open(res_txt_path, 'a')
        fp.write("===> loss: {}\n".format(loss))
        fp.write("===> icc: {}\n".format(icc))
        fp.write("===> mae: {}\n".format(mae))
        fp.write("===> average icc: {}\n".format(np.mean(icc)))
        fp.write("===> average mae: {}\n".format(np.mean(mae)))
        fp.close()
        print("icc {} \nmae {}".format(icc, mae))
        print("average icc {}\n".format(np.mean(icc)))
        print("average mae {}\n".format(np.mean(mae)))

        train_icc = np.mean(icc)
        train_mae = np.mean(mae)

        self.train_logger.add_scalar('loss', self.epoch_info['mean_loss'],
                                     self.meta_info['epoch'])
        self.train_logger.add_scalar('train_icc', train_icc,
                                     self.meta_info['epoch'])
        self.train_logger.add_scalar('train_mae', train_mae,
                                     self.meta_info['epoch'])

        for k, v in self.loss.items():
            self.iter_info[k] = loss_dict[k].data.item()
            self.train_logger.add_scalar(f'loss_{k}', np.mean(loss_value[k]),
                                         self.meta_info['epoch'])

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = dict()
        loss_dict = dict()
        result_frag = []
        label_frag = []

        print('validation dataloder length: ', len(loader))
        for image, label, state in loader:

            # get data
            image = image.float().to(self.dev)
            label = label.float().to(self.dev)
            state = state.float().to(self.dev)

            # inference
            with torch.no_grad():
                output, _, _ = self.model(image)

            if isinstance(output, list):
                result_frag.append(
                    (output[0] / (output[0] + output[1])).data.cpu().numpy())
            else:
                result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

        self.show_epoch_info()
        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        # compute metrics
        icc, mae, val_icc, val_mae = funcs.record_intensity_metrics(
            self.result, self.label, self.epoch_info['mean_loss'],
            self.arg.model_args['num_class'], self.arg.work_dir, 'val')

        torch.save(self.model.state_dict(),
                    os.path.join(self.arg.work_dir, 'final_model.pt'))

        self.validation_logger.add_scalar('val_icc', val_icc,
                                          self.meta_info['epoch'])
        self.validation_logger.add_scalar('val_mae', val_mae,
                                          self.meta_info['epoch'])

        for au in range(self.arg.model_args['num_class']):
            self.validation_logger.add_scalar(f'val_icc_{au}', icc[au],
                                              self.meta_info['epoch'])
            self.validation_logger.add_scalar(f'val_mae_{au}', mae[au],
                                              self.meta_info['epoch'])

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='TAS')

        # region arguments yapf: disable
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--lr_backbone', type=float, default=0.01, help='initial learning rate for backbone')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--lr_decay', type=float, default=0.1, help='lr decay for optimizer')
        parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        parser.add_argument('--backbone_only', type=str2bool, default=True, help='only use backbone weights')
        parser.add_argument('--pretrain', type=str2bool, default=True, help='load pretrained weights on ImageNet or not')
        parser.add_argument('--warmup_epoch', type=int,
                            default=-1, help='warmup epoch')

        parser.add_argument('--mixup_seg', type=bool, default=False, help='mixup within a segment or not')
        parser.add_argument('--alpha', type=float, default=0.2, help='mixup interpolation coefficient')

        # loss
        parser.add_argument('--loss', default=None,
                            help='the loss will be used')
        parser.add_argument('--loss_args', action=DictAction,
                            default=dict(), help='the arguments of loss')

        # scheduler
        parser.add_argument('--scheduler', default='constant',
                            help='the scheduler will be used')
        parser.add_argument('--scheduler_args', action=DictAction,
                            default=dict(), help='the arguments of scheduler')
        # endregion yapf: enable

        return parser
