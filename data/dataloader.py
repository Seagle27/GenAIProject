import torchaudio
from torch.utils.data import Dataset
import cv2
import PIL
import random
import numpy as np
from packaging import version
from PIL import Image
import os
import torch
import pandas as pd


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------
imagenet_templates_small = [
    "a photo of {}"
]


class VGGSound(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            logger,
            size=512,
            interpolation='bicubic',
    ):

        self.base_dir = args.data_dir
        self.vggsound = args.csv_path
        self.audio_path, self.image_path, self.label, self.num_label = list(), list(), list(), list()
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = args.placeholder_token
        self.input_length = args.input_length
        self.df = pd.read_csv(self.vggsound)
        self.data_set = args.data_set
        if self.data_set == 'train':
            self.center_crop = args.center_crop
            self.filter_frames = args.filter_frames
        else:
            self.center_crop = False
            self.filter_frames = False

        self.unique_labels = self.df['label'].unique().tolist()
        audios, images = set(), set()
        for label_name in self.unique_labels:
            label_audio_dir = os.path.join(args.data_dir, label_name, 'audio')
            label_image_dir = os.path.join(args.data_dir, label_name, 'image')
            audios = audios | set([file_path[:-4] for file_path in os.listdir(label_audio_dir)])
            images = images | set([file_path[:-4] for file_path in os.listdir(label_image_dir)])

        samples = audios & images if self.data_set == 'train' else audios
        self.prepare_dataset(samples)

        self.num_samples = len(self.audio_path)
        self._length = self.num_samples
        logger.info(f"{args.data_set}, num samples: {self.num_samples}")

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.templates = imagenet_templates_small

    def __len__(self):
        return self._length

    def prepare_dataset(self, samples):
        label_to_idx = {label: i for i, label in enumerate(self.unique_labels)}
        df = self.df[self.df["train/test split"] == self.data_set]
        for _, row in df.iterrows():
            file_name = f"{row['YouTube ID']}_{row['start seconds']:06d}"
            if file_name in samples:
                self.audio_path.append(os.path.join(self.base_dir, row["label"], 'audio', file_name + ".wav"))
                self.image_path.append(os.path.join(self.base_dir, row["label"], 'image', file_name + ".jpg"))
                self.label.append(row["label"])
                self.num_label.append(label_to_idx[row["label"]])

    def img_proc(self, img_path):
        image = cv2.imread(img_path)
        img = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (h,w,) = (img.shape[0],img.shape[1],)
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]
        image = Image.fromarray(img)
        image = image.resize((512, 320), resample=self.interpolation)
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)

    def aud_proc_beats(self, aud, rand_sec=0):
        wav, sr = torchaudio.load(aud)
        wav = torch.tile(wav, (1, 10))
        wav = wav[:, :sr*10]
        start = rand_sec * sr
        end = (rand_sec+self.input_length) * sr
        wav = wav[:, start:end]
        return wav[0]

    def txt_proc(self):
        text = random.choice(self.templates).format(self.placeholder_token)
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __getitem__(self, i):
        rand_sec = 0 if self.input_length == 10 else np.random.randint(0, 10 - self.input_length)
        example = {}
        aud = self.audio_path[i % self.num_samples]
        img = self.image_path[i % self.num_samples]
        example["input_ids"] = self.txt_proc()
        example['label'] = self.label[i % self.num_samples]
        example["audio_values"] = self.aud_proc_beats(aud, rand_sec)
        example['num_label'] = self.num_label[i % self.num_samples]
        example["pixel_values"] = self.img_proc(img) if self.data_set == "train" else 0
        example["full_name"] = aud.split('/')[-1][:-4]
        return example
