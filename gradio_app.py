import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from pydub import AudioSegment
import re
import os
from datetime import datetime
import gradio as gr

# Define the function to generate audio based on a prompt
def generate_audio(prompt, steps, cfg_scale, sigma_min, sigma_max, generation_time, seed, sampler_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download model
    model, model_config = get_pretrained_model("audo/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)

    # Set up text and timing conditioning
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": generation_time
    }]

    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
        seed=seed
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to temporary file
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save("temp_output.wav", output, sample_rate)

    # Convert to MP3 format using pydub
    audio = AudioSegment.from_wav("temp_output.wav")

    # Create Output folder and dated subfolder if they do not exist
    output_folder = "Output"
    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(output_folder, date_folder)
    os.makedirs(save_path, exist_ok=True)

    # Generate a filename based on the prompt
    filename = re.sub(r'\W+', '_', prompt) + ".mp3"  # Replace non-alphanumeric characters with underscores
    full_path = os.path.join(save_path, filename)

    # Ensure the filename is unique by appending a number if the file already exists
    base_filename = filename
    counter = 1
    while os.path.exists(full_path):
        filename = f"{base_filename[:-4]}_{counter}.mp3"
        full_path = os.path.join(save_path, filename)
        counter += 1

    # Export the audio to MP3 format
    audio.export(full_path, format="mp3")

    return full_path

def audio_generator(prompt, sampler_type, steps, cfg_scale, sigma_min, sigma_max, generation_time, seed):
    try:
        print("Generating audio with parameters:")
        print("Prompt:", prompt)
        print("Sampler Type:", sampler_type)
        print("Steps:", steps)
        print("CFG Scale:", cfg_scale)
        print("Sigma Min:", sigma_min)
        print("Sigma Max:", sigma_max)
        print("Generation Time:", generation_time)
        print("Seed:", seed)
        
        filename = generate_audio(prompt, steps, cfg_scale, sigma_min, sigma_max, generation_time, seed, sampler_type)
        return gr.Audio(filename), f"Generated: {filename}"
    except Exception as e:
        return str(e)

# Create Gradio interface
prompt_textbox = gr.Textbox(lines=5, label="Prompt")
sampler_dropdown = gr.Dropdown(
    label="Sampler Type",
    choices=[
        "dpmpp-3m-sde",
        "dpmpp-2m-sde",
        "k-heun",
        "k-lms",
        "k-dpmpp-2s-ancestral",
        "k-dpm-2",
        "k-dpm-fast"
    ],
    value="dpmpp-3m-sde"
)
steps_slider = gr.Slider(minimum=0, maximum=200, label="Steps", step=1)
steps_slider.value = 100  # Set the default value here
cfg_scale_slider = gr.Slider(minimum=0, maximum=15, label="CFG Scale", step=0.1)
cfg_scale_slider.value = 7  # Set the default value here
sigma_min_slider = gr.Slider(minimum=0, maximum=50, label="Sigma Min", step=0.1, value=0.3)
sigma_max_slider = gr.Slider(minimum=0, maximum=1000, label="Sigma Max", step=1, value=500)
generation_time_slider = gr.Slider(minimum=0, maximum=47, label="Generation Time (seconds)", step=1)
generation_time_slider.value = 47  # Set the default value here
seed_slider = gr.Slider(minimum=-1, maximum=999999, label="Seed", step=1)
seed_slider.value = 77212  # Set the default value here

output_textbox = gr.Textbox(label="Output")

title = "💀🔊 StableAudioWebUI 💀🔊"
description = "[Github Repository](https://github.com/Saganaki22/StableAudioWebUI)"

gr.Interface(
    audio_generator,
    [prompt_textbox, sampler_dropdown, steps_slider, cfg_scale_slider, sigma_min_slider, sigma_max_slider, generation_time_slider, seed_slider],
    [gr.Audio(), output_textbox],
    title=title,
    description=description
).launch()
