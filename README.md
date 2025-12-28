# 🦊 Spirit Animal Generator

An AI-powered application that analyzes your facial features and transforms your photo into your spirit animal using advanced machine learning models.

## 📋 Overview

This project uses a combination of computer vision and generative AI to:
1. Analyze facial features using GPT-4 Vision
2. Extract pose information using OpenPose
3. Generate spirit animal portraits using Stable Diffusion with ControlNet
4. Score and rank results using CLIP

## 🎯 Features

- **AI-Powered Animal Matching**: Uses GPT-4 Vision to analyze facial features and match them to animals from the entire animal kingdom
- **Pose-Preserving Generation**: Maintains your original pose using ControlNet OpenPose
- **Multiple Candidates**: Generates up to 5 candidate images and selects the best one
- **CLIP Scoring**: Automatically ranks generated images using CLIP similarity scores
- **User-Friendly Interface**: Clean Gradio web interface for easy interaction

## 🛠️ Tech Stack

- **Stable Diffusion v1.5** - Base image generation model
- **ControlNet OpenPose** - Pose-guided image generation
- **GPT-4 Vision** - Facial analysis and animal matching
- **CLIP** - Image-text similarity scoring
- **Gradio** - Web interface
- **PyTorch** - Deep learning framework

## ⚙️ Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster generation)
- Hugging Face account
- OpenAI API account

## 🔑 Required API Keys

> **Important**: This project requires the following API keys to function:

| API Key | Purpose | How to Get |
|---------|---------|------------|
| `HUGGINGFACE_TOKEN` | Access to Hugging Face models | [Create token here](https://huggingface.co/settings/tokens) |
| `OPENAI_API_KEY` | GPT-4 Vision for facial analysis | [Get API key here](https://platform.openai.com/api-keys) |

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/priyaljamin/5890_CVLords_Spirit_Animal.git
   cd 5890_CVLords_Spirit_Animal
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root or set environment variables:

   **Option A: Using .env file**
   ```
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   **Option B: Setting environment variables directly**

   On Windows (PowerShell):
   ```powershell
   $env:HUGGINGFACE_TOKEN="your_huggingface_token_here"
   $env:OPENAI_API_KEY="your_openai_api_key_here"
   ```

   On Mac/Linux:
   ```bash
   export HUGGINGFACE_TOKEN="your_huggingface_token_here"
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

## 🚀 Usage

1. **Run the application**
   ```bash
   python app.py
   ```

2. **Open the web interface**
   - The application will launch a Gradio interface
   - A local URL (e.g., `http://127.0.0.1:7860`) will be displayed
   - A public shareable link will also be generated

3. **Generate your spirit animal**
   - Upload a photo with a clear view of your face
   - Adjust settings (steps, guidance, seed, number of candidates)
   - Click "✨ Generate"
   - Wait for the AI to analyze and generate your spirit animal

## 📁 Project Structure

```
├── app.py              # Main Gradio application
├── baseline.py         # Stable Diffusion + ControlNet pipeline
├── gpt_map.py          # GPT-4 Vision facial analysis
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## ⚡ Performance

| Hardware | Time per Candidate |
|----------|-------------------|
| NVIDIA T4 GPU | ~15-25 seconds |
| CPU Only | ~2-5 minutes |

## 🎛️ Settings Guide

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Steps | 10-50 | 22 | More steps = higher quality but slower |
| Guidance | 4.0-12.0 | 7.5 | Higher = more prompt adherence |
| Seed | 0+ | 0 (random) | Set for reproducible results |
| Candidates | 1-5 | 3 | Number of images to generate |

## 🔧 Troubleshooting

**"No module named 'xxx'"**
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**"CUDA out of memory"**
- Reduce the number of candidates
- Lower the number of steps
- Use CPU mode (slower but works)

**"OpenAI API error"**
- Verify your `OPENAI_API_KEY` is set correctly
- Check your OpenAI account has available credits

**"Hugging Face authentication error"**
- Verify your `HUGGINGFACE_TOKEN` is set correctly
- Ensure your token has read access

## 👥 Authors

**CVLords Team** - Course 5810 Project

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis
- [ControlNet](https://github.com/lllyasviel/ControlNet) by lllyasviel
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Gradio](https://gradio.app/) for the web interface
