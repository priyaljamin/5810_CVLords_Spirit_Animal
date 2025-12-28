# app.py
import warnings
warnings.filterwarnings('ignore')

import gradio as gr
from PIL import Image
import os
from gpt_map import suggest_animal
from baseline import make_pose_map, build_pipeline, run_generation, score_image

from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_TOKEN"))

pipe = None
MAX_CANDIDATES = 5


def setup():
    global pipe
    if pipe is None:
        print("\n" + "=" * 60)
        print("INITIALIZING MODELS (this may take 1-2 minutes)...")
        print("=" * 60 + "\n")
        pipe = build_pipeline()
        print("\n" + "=" * 60)
        print("✓ ALL MODELS LOADED - Ready to generate!")
        print("=" * 60 + "\n")


def generate_spirit_animal(image, steps, guidance, seed, num_candidates):
    def empty_return(msg):
        return (msg, None, None) + tuple([None] * MAX_CANDIDATES)

    if image is None:
        return empty_return("❌ Please upload an image first!")

    try:
        setup()
        num_candidates = int(num_candidates)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        print("\n" + "=" * 60)
        print("PROCESSING YOUR IMAGE")
        print("=" * 60 + "\n")

        # Step 1: Analyze with GPT-4
        print("STEP 1/3: Analyzing face...")
        tmp_path = "temp_input.jpg"
        image.save(tmp_path, quality=95)
        mapping = suggest_animal(tmp_path)
        print(f"✓ Spirit Animal: {mapping['animal'].upper()}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Step 2: Extract pose
        print("\nSTEP 2/3: Extracting pose...")
        image_512 = image.resize((512, 512))
        pose = make_pose_map(image_512)

        # Step 3: Generate candidates
        print(f"\nSTEP 3/3: Generating {num_candidates} candidate(s)...")
        candidates = []
        for i in range(num_candidates):
            print(f"  Generating {i+1}/{num_candidates}...")
            img = run_generation(
                pipe, pose,
                f"{mapping['animal']} spirit-animal portrait, {mapping['prompt']}",
                mapping['negative'],
                steps, guidance,
                seed+i if seed else None
            )
            clip_prompt = f"{mapping['animal']} spirit-animal portrait"
            s = score_image(clip_prompt, img)
            candidates.append({"image": img, "clip_score": s})
            print(f"  ✓ Score: {s:.4f}")

        # Pick best
        best_idx = max(range(len(candidates)), key=lambda i: candidates[i]["clip_score"])
        spirit_animal = candidates[best_idx]["image"]
        print(f"\n✓ Selected candidate #{best_idx + 1}")
        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60 + "\n")

        # Build output text
        scores_table = "| Candidate | CLIP Score |\n|-----------|------------|\n"
        for i, c in enumerate(candidates):
            marker = "⭐ BEST" if i == best_idx else ""
            scores_table += f"| #{i+1} | {c['clip_score']:.4f} {marker} |\n"

        analysis_text = f"""
# 🦊 Your Spirit Animal: **{mapping['animal'].title()}**

### 📋 Why this animal?
{mapping['reason']}

### ⚙️ Settings
- Steps: {steps} | Guidance: {guidance} | Seed: {seed if seed else 'Random'} | Candidates: {num_candidates}

### 🏆 CLIP Scores
{scores_table}
"""

        # Pad to MAX_CANDIDATES
        candidate_images = [c["image"] for c in candidates]
        while len(candidate_images) < MAX_CANDIDATES:
            candidate_images.append(None)

        return (analysis_text, pose, spirit_animal,
                candidate_images[0], candidate_images[1], candidate_images[2],
                candidate_images[3], candidate_images[4])

    except Exception as e:
        import traceback
        return empty_return(f"❌ Error: {str(e)}\n\n{traceback.format_exc()}")


# Gradio UI
with gr.Blocks(title="Spirit Animal Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦊 Spirit Animal Generator\nUpload your photo to discover your spirit animal!")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📸 Upload Photo", height=350)
            gr.Markdown("### ⚙️ Settings")
            steps_slider = gr.Slider(10, 50, value=22, step=1, label="Steps")
            guidance_slider = gr.Slider(4.0, 12.0, value=7.5, step=0.5, label="Guidance")
            seed_input = gr.Number(value=0, label="Seed (0=random)")
            num_candidates_slider = gr.Slider(1, MAX_CANDIDATES, value=3, step=1, label="Candidates")
            generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            analysis_output = gr.Markdown("👆 Upload an image to start!")
            pose_output = gr.Image(label="🦴 Pose", height=200)
            spirit_output = gr.Image(label="🎨 Best Result", height=250)

    gr.Markdown("---\n## 🖼️ All Candidates")
    with gr.Row():
        c1 = gr.Image(label="#1", height=180)
        c2 = gr.Image(label="#2", height=180)
        c3 = gr.Image(label="#3", height=180)
    with gr.Row():
        c4 = gr.Image(label="#4", height=180)
        c5 = gr.Image(label="#5", height=180)

    gr.Markdown("---\n**⚡ With T4 GPU:** ~15-25 sec per candidate")

    generate_btn.click(
        fn=generate_spirit_animal,
        inputs=[input_image, steps_slider, guidance_slider, seed_input, num_candidates_slider],
        outputs=[analysis_output, pose_output, spirit_output, c1, c2, c3, c4, c5]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)