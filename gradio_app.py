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

# Define a function to toggle the visibility of the seed slider
def toggle_seed_slider(x):
    seed_slider.visible = not x

# Define a function to set up the model and device
def setup_model(model_half):
    model, model_config = get_pretrained_model("audo/stable-audio-open-1.0")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Convert model to float16 if model_half is True
    if model_half:
        model = model.to(torch.float16)
        print("Model data type:", next(model.parameters()).dtype)
    
    return model, model_config, device

# Define the function to generate audio based on a prompt
def generate_audio(prompt, steps, cfg_scale, sigma_min, sigma_max, generation_time, seed, sampler_type, model_half, model, model_config, device):
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
        sample_size=model_config["sample_size"],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
        seed=seed
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, and convert to int16 directly if model_half is used
    output = output.div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767)
    if model_half:
        output = output.to(torch.int16).cpu()
    else:
        output = output.to(torch.float32).to(torch.int16).cpu()

    torchaudio.save("temp_output.wav", output, model_config["sample_rate"])

    # Convert to MP3 format using pydub
    audio = AudioSegment.from_wav("temp_output.wav")

    # Create Output folder and dated subfolder if they do not exist
    output_folder = "Output"
    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(output_folder, date_folder)
    os.makedirs(save_path, exist_ok=True)

    # Set a maximum filename length (e.g., 50 characters)
    max_length = 50
    if len(prompt) > max_length:
        prompt = prompt[:max_length] + "_truncated"

    # Sanitize the prompt to create a safe filename
    filename = re.sub(r'\W+', '_', prompt) + ".mp3"
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

def audio_generator(prompt, sampler_type, steps, cfg_scale, sigma_min, sigma_max, generation_time, random_seed, seed, model_half):
    try:
        print("Generating audio with parameters:")
        print("Prompt:", prompt)
        print("Sampler Type:", sampler_type)
        print("Steps:", steps)
        print("CFG Scale:", cfg_scale)
        print("Sigma Min:", sigma_min)
        print("Sigma Max:", sigma_max)
        print("Generation Time:", generation_time)
        print("Random Seed:", "Random" if random_seed else "Fixed")
        print("Seed:", seed)
        print("Model Half Precision:", model_half)
        
        # Set up the model and device
        model, model_config, device = setup_model(model_half)
        
        if random_seed:
            seed = torch.randint(0, 1000000, (1,)).item()
        
        filename = generate_audio(prompt, steps, cfg_scale, sigma_min, sigma_max, generation_time, seed, sampler_type, model_half, model, model_config, device)
        return gr.Audio(filename), f"Generated: {filename}"
    except Exception as e:
        return str(e)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; font-size: 300%;'>💀🔊 StableAudioWebUI 💀🔊</h1>")

    # Main input components
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
    steps_slider = gr.Slider(minimum=0, maximum=200, label="Steps", step=1, value=100)
    generation_time_slider = gr.Slider(minimum=0, maximum=47, label="Generation Time (seconds)", step=1, value=47)
    random_seed_checkbox = gr.Checkbox(label="Random Seed")
    seed_slider = gr.Slider(minimum=-1, maximum=999999, label="Seed", step=1, value=123456)

    # Advanced parameters accordion
    with gr.Accordion("Advanced Parameters", open=False):
        cfg_scale_slider = gr.Slider(minimum=0, maximum=15, label="CFG Scale", step=0.1, value=7)
        sigma_min_slider = gr.Slider(minimum=0, maximum=50, label="Sigma Min", step=0.1, value=0.3)
        sigma_max_slider = gr.Slider(minimum=0, maximum=1000, label="Sigma Max", step=0.1, value=500)

    # Low VRAM checkbox and submit button
    model_half_checkbox = gr.Checkbox(label="Low VRAM (float16)", value=False)
    submit_button = gr.Button("Generate")

    # Define the output components
    audio_output = gr.Audio()
    output_textbox = gr.Textbox(label="Output")

    # Link the button and the function
    random_seed_checkbox.change(fn=toggle_seed_slider, inputs=[random_seed_checkbox], outputs=[seed_slider])
    submit_button.click(audio_generator,
                        inputs=[prompt_textbox, sampler_dropdown, steps_slider, cfg_scale_slider,sigma_min_slider, sigma_max_slider, generation_time_slider, random_seed_checkbox, seed_slider, model_half_checkbox],
                        outputs=[audio_output, output_textbox])

    # GitHub link at the bottom
    gr.Markdown("<p style='text-align: center;'><a href='https://github.com/Saganaki22/StableAudioWebUI'>Github Repository</a></p>")

# Launch the Gradio demo
demo.launch()
