import os
import argparse
import torch.utils.checkpoint
from diffusers import StableDiffusionPipeline, DDIMScheduler

from inference import inference
from ptp.prompt_to_prompt import *
from ptp.null_txt_inversion import *
from ptp.ptp_consts import *


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a image editing script.")
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
    parser.add_argument("--image_prompt", type=str, default='a cat sitting next to a mirror')
    parser.add_argument("--edit_prompt", type=str, default='a <*> sitting next to a mirror')
    parser.add_argument("--image_path", type=str, default=None)
    # parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--input_length", type=int, default=10,
                        help="Select the number of seconds of audio you want in each test-sample.")
    parser.add_argument("--lora", type=bool, default=False,
                        help="Whether load Lora layers or not")
    parser.add_argument("--eval_mode", type=bool, default=False,
                        help="Whether save ground truth frame or not")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        start_time=50,
        return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)

    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def run_and_display(ldm, prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None,
                    verbose=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS,
                                        guidance_scale=GUIDANCE_SCALE, generator=generator,
                                        uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t


def edit_img_loop(args, tokenizer, accelerator, model, weight_dtype, test_dataloader):
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    accelerator.unwrap_model(model).text_encoder.resize_token_embeddings(len(tokenizer))

    edit_prompt = args.edit_prompt
    image_label = args.image_prompt

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)

    ldm_model = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(model).text_encoder,
        vae=accelerator.unwrap_model(model).vae,
        unet=accelerator.unwrap_model(model).unet,
        scheduler=scheduler
    ).to(accelerator.device)
    null_inversion = NullInversion(ldm_model)
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(args.image_path, image_label,
                                                                          offsets=(0, 0, 200, 0), verbose=True)

    prompts = [image_label]
    controller = AttentionStore()
    _, x_t = run_and_display(ldm_model, prompts, controller, run_baseline=False, latent=x_t,
                             uncond_embeddings=uncond_embeddings, verbose=False)

    for step, batch in enumerate(test_dataloader):
        if step >= args.generation_steps:
            break
        # Audio's feature extraction
        audio_values = batch["audio_values"].to(accelerator.device).to(dtype=weight_dtype)
        aud_features = accelerator.unwrap_model(model).aud_encoder.extract_features(audio_values)[1]
        audio_token = accelerator.unwrap_model(model).embedder(aud_features)

        token_embeds = model.text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = audio_token.clone()

        ldm_model = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=accelerator.unwrap_model(model).text_encoder,
            vae=accelerator.unwrap_model(model).vae,
            unet=accelerator.unwrap_model(model).unet,
            scheduler=scheduler
        ).to(accelerator.device)

        cross_replace_steps = {'default_': .8, }
        self_replace_steps = .5
        blend_word = (('cat',), (args.placeholder_token,))
        eq_params = {"words": (args.placeholder_token,), "values": (2,)}  # amplify attention to the word learned audio token by 2

        controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps, blend_word, eq_params)
        images, _ = run_and_display(ldm_model, prompts, controller, run_baseline=False, latent=x_t,
                                    uncond_embeddings=uncond_embeddings)


if __name__ == '__main__':
    args = parse_args()
    tokenizer, accelerator, model, _, weight_dtype, test_data = inference(args)
    edit_img_loop(args, tokenizer, accelerator, model, weight_dtype, test_data)
