# sys
import os
import sys
import numpy as np
import random
import pickle

# image preprocess
import cv2
from feeder.randaugment import RandAugment
from PIL import Image

# torch
import torch
from torchvision import transforms


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        label_path: the path to label
        image_path: the path to image
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 label_path,
                 image_path,
                 trend_path=None,
                 state_path=None,
                 debug=False,
                 image_size=256,
                 istrain=False,
                 imagepath=False,
                 sampling_strategy='random',
                 repeat_ok=False,
                 smooth=False,
                 frames_between_keyframe=0,
                 isaug=False,
                 return_short_mark=False,
                 **kwargs):

        self.debug = debug
        self.label_path = label_path
        self.image_path = image_path
        self.trend_path = trend_path
        self.state_path = state_path
        self.image_size = image_size
        self.istrain = istrain
        self.imagepath = imagepath
        self.sampling_strategy = sampling_strategy
        self.frames_between_keyframe = frames_between_keyframe
        self.repeat_ok = repeat_ok
        self.smooth = smooth
        self.isaug = isaug

        self.return_short_mark = return_short_mark
        
        self.segments = list()
        self.rand_aug = transforms.Compose([
            RandAugment(1, 9),
            transforms.Resize((self.image_size, self.image_size)),
        ])

        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    
        self.load_data()

    def load_segments(self):

        def select_frame(seg_idx, num_interp_frames):
            res = []
            if len(seg_idx) < num_interp_frames + 2:
                if self.repeat_ok:
                    loc_idx = random.sample(list(range(1, num_interp_frames + 1)), len(seg_idx) - 1)
                    loc_idx.sort()
                    frame_idx = np.zeros(num_interp_frames + 2).astype(np.int)
                    pos = 0
                    for loc in loc_idx:
                        for i in range(pos, loc + 1):
                            frame_idx[i] = loc_idx.index(loc)
                        pos = loc + 1
                    if pos < num_interp_frames + 2:
                        for i in range(pos, num_interp_frames + 2):
                            frame_idx[i] = len(seg_idx) - 1
                    res = [seg_idx[i] for i in frame_idx]
                    return res
                else:
                    return res
            if self.sampling_strategy == 'random':
                frame_idx = random.sample(list(range(1, len(seg_idx) - 1)), num_interp_frames)
                frame_idx.sort()
            elif self.sampling_strategy == 'linear':
                frame_idx = np.linspace(0,
                                        len(seg_idx) - 1,
                                        num_interp_frames + 2,
                                        dtype=np.uint8)[1:-1]
            res = [seg_idx[i] for i in frame_idx]
            res = [seg_idx[0]] + res + [seg_idx[-1]]
            return res

        cnt_short = 0
        short_mark = []
        for au in range(self.num_classes):
            # labels = self.labels[:, au]
            labels = self.labels[:]
            states = self.states[:, au]
            trends = self.trends[:, au]
            img_path = self.imagepaths

            key_idx_list = []
            for idx, (trend, state) in enumerate(zip(trends, states)):
                if trend != 0 and state != 0:
                    key_idx_list.append(idx)

            for k_idx in range(1, len(key_idx_list)):
                start_idx = key_idx_list[k_idx - 1]
                end_idx = key_idx_list[k_idx]
                if ''.join(img_path[start_idx].split('/')[:-1]) != ''.join(
                        img_path[end_idx].split('/')[:-1]):
                    continue

                if self.frames_between_keyframe > 0:
                    if len(list(range(start_idx, end_idx + 1))) < self.frames_between_keyframe + 2:
                        cnt_short += 1
                        short_mark.append(True)
                    else:
                        short_mark.append(False)
                    res = select_frame(list(range(start_idx, end_idx + 1)),
                                       self.frames_between_keyframe)

                key_seg = []
                for n, f_idx in enumerate(res):
                    if self.smooth:
                        smoothed_label = labels[res[-1]] * (
                            n / (len(res) - 1)) + labels[res[0]] * (
                                1 - (n / (len(res) - 1)))
                        key_seg.append({
                            'img_path': img_path[f_idx],
                            'label': smoothed_label,
                            'state': states[f_idx],
                            'trend': trends[f_idx],
                            'au': au
                        })
                    else:
                        key_seg.append({
                            'img_path': img_path[f_idx],
                            'label': labels[f_idx],
                            'state': states[f_idx],
                            'trend': trends[f_idx],
                            'au': au
                        })

                if len(key_seg) == self.frames_between_keyframe + 2:
                    # print(key_seg)
                    self.segments.append(key_seg)

        if self.repeat_ok:
            self.short_mark = short_mark
        print('num of short segments: ', cnt_short)

    # all segment evolves from valley to peak
    def segment_change_order(self):
        for idx in range(len(self.segments)):
            seg = self.segments[idx]
            if seg[0]['label'][seg[0]['au']] > seg[-1]['label'][seg[0]['au']]:
                seg.reverse()
                self.segments[idx] = seg

    def load_data(self):
        # data: N C H W

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.labels = pickle.load(f)

        # load image
        with open(self.image_path, 'rb') as f:
            self.sample_name, self.imagepaths = pickle.load(f)

        # load trend and state
        if self.state_path is not None:
            with open(self.state_path, 'rb') as f:
                self.sample_name, self.states = pickle.load(f)
        if self.trend_path is not None:
            with open(self.trend_path, 'rb') as f:
                self.sample_name, self.trends = pickle.load(f)

        self.labels = np.array(self.labels).squeeze()
        self.imagepaths = np.array(self.imagepaths).squeeze()
        self.states = np.array(self.states).squeeze()
        self.trends = np.array(self.trends).squeeze()

        if self.debug:
            self.labels = self.labels[0:100]
            self.imagepaths = self.imagepaths[0:100]
            self.states = self.states[0:100]
            self.trends = self.trends[0:100]

        self.num_frames = len(self.labels)
        self.num_classes = self.labels.shape[-1]

        if self.istrain:
            self.load_segments()
            self.segment_change_order()
            random.shuffle(self.segments)
        else:
            for f_idx in range(self.num_frames):
                self.segments.append({
                    'img_path': self.imagepaths[f_idx],
                    'label': self.labels[f_idx],
                    'state': self.states[f_idx],
                    'trend': self.trends[f_idx],
                })

        print('num of valid segments: ', len(self.segments))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        segment = self.segments[index]
        label = []
        state = []
        trend = []
        image = []
        img_path = []
        # val
        if isinstance(segment, dict):
            label = segment['label']
            state = segment['state']
            trend = segment['trend']
            img = cv2.imread(segment['img_path'].replace(
                "../../..", "../..", 1))

            face = cv2.resize(img, (self.image_size, self.image_size))
            face = face.transpose((2, 0, 1))
            image.append(face / 255.0)
            img_path.append(segment['img_path'])
        # train
        else:
            for seg_idx in range(len(segment)):
                label.append(segment[seg_idx]['label'])
                state.append(segment[seg_idx]['state'])
                trend.append(segment[seg_idx]['trend'])

                if self.isaug:
                    face = Image.open(segment[seg_idx]['img_path'].replace(
                        "../../..", "../..", 1))
                    face = self.rand_aug(face)
                    face = np.array(face).transpose((2, 0, 1))
                    image.append(face / 255.0)
                else:
                    img = cv2.imread(segment[seg_idx]['img_path'].replace(
                        "../../..", "../..", 1))

                    face = cv2.resize(img, (self.image_size, self.image_size))
                    face = face.transpose((2, 0, 1))
                    image.append(face / 255.0)
                img_path.append(segment[seg_idx]['img_path'])
        label = np.array(label)
        state = np.array(state)
        trend = np.array(trend)
        image = np.array(image)

        if self.istrain:
            au = segment[0]['au']
            if self.return_short_mark:
                short_mark = self.short_mark[index]
                return image, label, state, au, short_mark
            else:
                return image, label, state, au
        elif self.imagepath:
            return image, label, state, trend, img_path
        else:
            return image, label, state
