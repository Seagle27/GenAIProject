# SoundVision: Audio-Driven Image Generation with Cross-Modal Alignment
Final course project in the course "Generative AI" in BGU

This repo is mainly based, and forked from AudioToken's github repo([*AudioToken: Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation*](https://pages.cs.huji.ac.il/adiyoss-lab/AudioToken/))

# Abstract
Recent advancements in text-to-image generation have demonstrated remarkable success in

synthesizing visually coherent images from natural language descriptions. However, audio-to-
image generation remains a less-explored and challenging task due to the fundamental differences

between audio and visual modalities. Audio signals encode rich temporal and spectral information,
making it difficult to directly map them to structured spatial representations in an image. Existing
models such as AudioToken attempt to bridge this gap, but struggle with semantic alignment and
generalization.
This project aims to enhance the the AudioToken framework for audio-driven image generation,
by improving cross-modal alignment. We introduce a contrastive learning-based framework to
enhance audio-to-image generation. Our method incorporates InfoNCE-based contrastive losses
to refine audio-text alignment and structure the audio embedding space. Specifically, we introduce
SinglePositiveInfoNCE loss for improved cross-modal alignment and
AudioAudioInfoNCESymmetric loss to enhance intra-modal clustering of audio features.
Additionally, we showcase ability of guided diffusion-based image editing, allowing fine-grained
modifications to images based on audio cues.
We evaluate our method on a filtered subset of the VGGSound dataset, comparing it against
AudioToken baselines. Our model demonstrates an improvement in zero-shot classification
accuracy and maintains competitive performance in standard retrieval metrics. These results
indicate that InfoNCE-based loss functions contribute to better cross-modal alignment and
generalization.

# Installation
```
git clone https://github.com/Seagle27/GenAIProject.git
cd GenAIProject
pip install -r requirements.txt
```
Configure your Accelerate environment with:
```angular2html
accelerate config
```

Download BEATs pre-trained model:

[Download Link](https://onedrive.live.com/?authkey=%21APLo1x9WFLcaKBI&id=6B83B49411CA81A7%2125955&cid=6B83B49411CA81A7&parId=root&parQt=sharedby&o=OneUp)


# Training

First, download our data set. [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/). Download links for the dataset can be found [here](https://huggingface.co/datasets/Loie/VGGSound/tree/main/).

```angular2html
!accelerate launch train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --csv_path=$CSV_PATH \
    --resolution=512 \
    --train_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=17250 \
    --learning_rate=8.0e-05 \
    --save_steps=1000 \
    --pretrained_audio_encoder_path=$BEATS_PATH 
```

# Inference

After you've trained a model with the above command, you can generate images using the following script:
```angular2html
!accelerate launch inference.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --learned_embeds=$LEARNED_EMBEDS \
  --pretrained_audio_encoder_path=$BEATS_PATH \
  --csv_path=$CSV_PATH \
  --generation_steps=100 
```