import gradio as gr
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os
from huggingface_hub import hf_hub_download

# ---------- CONFIG ----------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
# Change this to your Hugging Face Hub repository where you will upload your LoRA
LORA_FLICKER_HUB_REPO = "Khatijaliya/flicker8k-lora"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Fixed inference parameters for optimal performance
INFERENCE_STEPS = 8
GUIDANCE_SCALE = 5

# Global variable for the pipeline
active_pipeline = None

# ---------- MODEL LOADING FUNCTION ----------
def load_model():
    """Load the base model and fine-tuned LoRA weights"""
    global active_pipeline
    
    print("Loading base Stable Diffusion model...")
    try:
        # Load the base model
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch_dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Load LoRA weights from the Hub
        print(f"Loading LoRA weights from {LORA_FLICKER_HUB_REPO}...")
        pipe.load_lora_weights(LORA_FLICKER_HUB_REPO)
        
        # Use a faster scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        pipe.to(device)
        
        # Enable memory optimizations
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing(1)
            print("✅ Attention slicing enabled")
        
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
            print("✅ VAE slicing enabled")
        
        # Enable xFormers if available
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ xFormers memory efficient attention enabled")
            except Exception as e:
                print(f"⚠️ xFormers not available: {e}")
        
        active_pipeline = pipe
        print(f"✅ Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# ---------- GENERATION FUNCTION ----------
def generate_image(prompt, seed, progress=gr.Progress()):
    """Generate image using the active pipeline with optimized settings"""
    
    if active_pipeline is None:
        return None, "❌ Model not loaded! Please reload the app."
    
    if not prompt or prompt.strip() == "":
        return None, "❌ Please enter a prompt!"
    
    try:
        progress(0.1, desc="Preparing generation...")
        
        # Set up generator
        if seed == -1:
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        
        progress(0.3, desc="Generating image...")
        
        # Generate with fixed optimized settings
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    result = active_pipeline(
                        prompt=prompt,
                        num_inference_steps=INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=generator,
                        return_dict=True
                    )
            else:
                result = active_pipeline(
                    prompt=prompt,
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                    return_dict=True
                )
        
        progress(1.0, desc="Complete!")
        
        info = f"✅ Generated successfully!\nOptimized Settings - Steps: {INFERENCE_STEPS}, Guidance: {GUIDANCE_SCALE}"
        if seed != -1:
            info += f", Seed: {seed}"
            
        return result.images[0], info
        
    except Exception as e:
        return None, f"❌ Generation failed: {str(e)}"

# ---------- BUILD UI ----------
def create_interface():
    """Create the Gradio interface with a single generation page"""
    
    with gr.Blocks(title="Flickr8k Image Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Flicker8k Image Generator")
        gr.Markdown("Fine-tuned on the Flicker8k dataset to generate images of people, animals, and outdoor scenes.")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="📝 Describe the image you want",
                    placeholder="e.g., A person walking a dog on a path, high quality photograph",
                    lines=3
                )
                seed = gr.Number(
                    value=-1, 
                    label="🎲 Seed (-1 for random)",
                    info="Use the same seed to reproduce results"
                )
                btn_generate = gr.Button("🎨 Generate Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_img = gr.Image(label="Generated Image", type="pil")
                generation_info = gr.Markdown("")
        
        btn_generate.click(
            generate_image,
            inputs=[prompt, seed],
            outputs=[output_img, generation_info]
        )
        
    return demo

# ---------- MAIN ----------
if __name__ == "__main__":
    print("🚀 Starting Flicker8k Image Generator...")
    
    # Initialize model
    if not load_model():
        print("❌ Failed to load model. The app cannot run.")
        exit(1)
    
    # Create and launch interface
    demo = create_interface()
    demo.launch()
