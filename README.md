# StableAudioWebUI

### A Lightweight Gradio Web interface for running Stable Audio Open 1.0 
<br>
<br>

![image](assets/screenshot.png)

<br>
<br>

---

  Start by cloning the repo:
 
    git clone https://github.com/Saganaki22/StableAudioWebUI.git

    
<br>
  Use the below deployment (tested on 24GB Nvidia VRAM):

    cd StableAudioWebUI
    conda create -n saowebui python=3.10
    conda activate saowebui
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt


    
<br>
  (Note if you have an older Nvidia GPU you may need to use CUDA 11.8)

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

<br>
  If you haven't got a hugging face account or have not used huggingface-cli before, create an account and then authenticate your Hugging face account with a token (create token at https://huggingface.co/settings/tokens)

    huggingface-cli login

  (paste your token and follow the instructions, token will not be displayed when pasted)

  ## ⚠ If you want to run it using CPU <br> 
  skip 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121' and just run

    pip install -r requirements.txt
    pip install -r requirements1.txt

##
Run

    python gradio_app.py
    
<br>

# Screenshots

(All with random seeds) <br>

Prompt: a dog barking <br>
CFG: 7 <br>
Sigma_Min: 0.3 <br>
Sigma_Max: 500 <br>

![image](https://github.com/Saganaki22/StableAudioWebUI/blob/main/assets/screenshot1.png)

#
<br>
Prompt: people clapping <br>
CFG: 7 <br>
Sigma_Min: 0.3 <br>
Sigma_Max: 500 <br>

![image](https://github.com/Saganaki22/StableAudioWebUI/blob/main/assets/screenshot2.png)

#
<br>
Prompt: didgeridoo <br>
CFG: 7 <br>
Sigma_Min: 0.3 <br>
Sigma_Max: 500 <br>

![image](https://github.com/Saganaki22/StableAudioWebUI/blob/main/assets/screenshot3.png)

## Model Details

- **Model type**: `Stable Audio Open 1.0` is a latent diffusion model based on a transformer architecture.
- **Language(s)**: English
- **License**: See the [LICENSE file](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE).
- **Commercial License**: to use this model commercially, please refer to [https://stability.ai/membership](https://stability.ai/membership)
