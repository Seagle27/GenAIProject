import argparse
import logging
import os
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from diffusers.utils import check_min_version
from scipy.io.wavfile import write

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from transformers import CLIPTokenizer

from data.dataloader import VGGSound
from modules.AudioToken.AudioToken import AudioTokenWrapper

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.12.0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--learned_embeds", type=str, default='output/embedder_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--learned_embeds_lora", type=str, default='output/lora_layers_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2',
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--pretrained_audio_encoder_path", type=str,
                        default='models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt', help="Path to "
                                                                                                     "pretrained "
                                                                                                     "audio encoder"
                                                                                                     "(BEATs) model")
    parser.add_argument("--revision", type=str, default=None, required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to the VGGSound csv file.")
    parser.add_argument("--data_dir", type=str,
                        help="A folder containing the training data.")
    parser.add_argument("--placeholder_token", type=str, default="<*>",
                        help="A token to use as a placeholder for the audio.", )
    parser.add_argument("--output_dir", type=str, default="output",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=768,
                        help="The resolution for input images, all the images in the train/validation"
                             " dataset will be resized to this resolution")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of subprocesses to use for data loading."
                             " 0 means that the data will be loaded in the main process.")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
                             " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."
                             " Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
                             " For more information, see"
                             " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                             ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--data_set", type=str, default='test', choices=['train', 'test'],
                        help="Whether use train or test set")
    parser.add_argument("--generation_steps", type=int, default=50)
    parser.add_argument("--run_name", type=str, default='AudioToken',
                        help="Insert run name")
    parser.add_argument("--set_size", type=str, default='full')
    parser.add_argument("--prompt", type=str, default='a photo of <*>, 4k, high resolution')
    parser.add_argument("--input_length", type=int, default=10,
                        help="Select the number of seconds of audio you want in each test-sample.")
    parser.add_argument("--lora", type=bool, default=False,
                        help="Whether load Lora layers or not")
    parser.add_argument("--test_batch_size", type=int, default=1,
                        help="Batch size (per device) for the training dataloader.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


def inference(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    imgs_save_path = os.path.join(args.output_dir, 'image')
    if not os.path.exists(imgs_save_path):
        os.makedirs(imgs_save_path)

    audio_save_path = os.path.join(args.output_dir, 'audio')
    if not os.path.exists(audio_save_path):
        os.makedirs(audio_save_path)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    print(f"SEED: {args.seed}")

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    at_model = AudioTokenWrapper(args, accelerator).to(weight_dtype).eval()

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    test_dataset = VGGSound(
        args=args,
        tokenizer=tokenizer,
        logger=logger,
        size=args.resolution,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Prepare everything with our `accelerator`.
    at_model, test_dataloader = accelerator.prepare(
        at_model, test_dataloader
    )

    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    accelerator.unwrap_model(at_model).text_encoder.resize_token_embeddings(len(tokenizer))

    # Instead of instantiating the pipeline in every loop iteration, create it once here.
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(at_model).text_encoder,
        vae=accelerator.unwrap_model(at_model).vae,
        unet=accelerator.unwrap_model(at_model).unet,
    ).to(accelerator.device)

    # Optional: enable more memory‑efficient attention if available.
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pipeline.enable_attention_slicing("auto")

    prompt = args.prompt

    for step, batch in enumerate(test_dataloader):
        print(f"Generating Image {step + 1}/{len(test_dataloader)}")
        if step >= args.generation_steps:
            break
        save_path = os.path.join(imgs_save_path, f'{batch["full_name"][0]}_{batch["label"][0]}.png')
        if os.path.isfile(save_path):
            # print(f"Skipping {save_path}")
            continue
        # Audio's feature extraction
        with torch.cuda.amp.autocast(dtype=torch.float32):
            audio_values = batch["audio_values"].to(accelerator.device).to(dtype=weight_dtype)
            aud_features = accelerator.unwrap_model(at_model).aud_encoder.extract_features(audio_values)[1].to(
                dtype=weight_dtype)
        audio_token = accelerator.unwrap_model(at_model).embedder(aud_features)

        token_embeds = at_model.text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = audio_token.clone()

        pipeline.unet.set_attention_slice(None)
        pipeline._last_prompt = None
        image = pipeline(prompt, num_inference_steps=args.num_inference_steps, guidance_scale=7.5).images[0]
        image.save(save_path)

        # Extract audio values as a NumPy array
        audio_numpy = audio_values[0].to(dtype=torch.float32).cpu().numpy()  # Assuming batch dimension is 1

        # Define the sample rate (modify as needed)
        sample_rate = 16000  # Typical sample rate for audio processing

        # Save the .wav file
        audio_filename = os.path.join(audio_save_path, f'{batch["full_name"][0]}_{batch["label"][0]}.wav')
        write(audio_filename, sample_rate, audio_numpy)


if __name__ == "__main__":
    args = parse_args()
    inference(args)