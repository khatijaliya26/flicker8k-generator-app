
### **Flicker8k Image Generator**

This project demonstrates how to build a custom text-to-image model by fine-tuning the powerful Stable Diffusion model on the Flicker8k dataset. The final application is a user-friendly, interactive demo built with **Gradio** and deployed on **Hugging Face Spaces**.

### **UI**
<img width="1581" height="556" alt="image" src="https://github.com/user-attachments/assets/ad1ef742-afd8-480f-a29f-f38a38791e6d" />


### **Live Demo**

You can try out the live application on Hugging Face Spaces:

**[Live Demo Link](https://huggingface.co/spaces/Khatijaliya/flicker8k-generator)**

### **How It Works**

The core of this project is a **Stable Diffusion v1.5** model that has been fine-tuned using **LoRA (Low-Rank Adaptation)**. This technique allows us to adapt the model to a specific theme without training the entire multi-billion parameter model from scratch.

The entire workflow, from data handling to deployment, is contained within the `GenAI_Text_to_Image_Generator_Flikr.ipynb` notebook. The process is divided into three key stages:

1.  **Data Preparation**: The Flicker8k dataset, which contains over 40,000 image-caption pairs, is processed to create a structured `metadata.csv` file that links each image to its captions.
2.  **Fine-Tuning with LoRA**: A pre-trained Stable Diffusion model is efficiently fine-tuned on this data using the LoRA method. The output of this stage is the fine-tuned model weights saved in two files: **`adapter_model.safetensors`** and **`adapter_config.json`**.
3.  **Deployment**: The application is packaged and deployed to Hugging Face Spaces, where it uses the fine-tuned model to generate images from text prompts in real time.

### **Files and Folders**

  * `GenAI_Text_to_Image_Generator_Flikr.ipynb`: This Jupyter notebook contains all the code for the project, from data preparation and fine-tuning to the final deployment steps.
  * `app.py`: The main application script. It contains the Gradio UI and all the code for loading the model, handling user inputs, and generating images.
  * `requirements.txt`: This file lists all the necessary Python libraries that must be installed for the application to run, both locally and on Hugging Face Spaces.
  * `model/` (or your chosen path): The local directory where your fine-tuned LoRA files (`adapter_model.safetensors` and `adapter_config.json`) are saved after training. **These files are not uploaded to GitHub.** Instead, they are hosted on a Hugging Face model repository.

### **Project Setup and Deployment**

To run this project or deploy it yourself, follow these steps in a Google Colab notebook.

#### **Step 1: Get Your Hugging Face API Token**

You will need an API token with **write access** to create and upload files to your Space.

1.  Log in to your Hugging Face account.
2.  Go to your **Settings** page.
3.  Click on **Access Tokens** in the left-hand menu.
4.  Click the **New token** button.
5.  Give your token a name (e.g., `my-deployment-token`) and set the **Role** to `write`.
6.  Click **Generate token** and copy the value immediately. **You won't be able to see it again.**

#### **Step 2: Save Your LoRA Weights to Hugging Face Hub**

Your fine-tuned model weights are not in this GitHub repository. You must host them on the Hugging Face Hub.

1.  Create a new model repository on Hugging Face (e.g., `your-username/flicker8k-lora`).
2.  In your Colab notebook, use the `huggingface_hub` library or the web interface to upload your LoRA files (`adapter_model.safetensors` and `adapter_config.json`) to this new repository.

#### **Step 3: Prepare for Deployment**

In your Colab notebook, use the `%%writefile` magic command to create the `app.py` and `requirements.txt` files.

  * **Crucially**, update the `LORA_FLICKER_HUB_REPO` variable in `app.py` to point to the repository where you saved your LoRA weights. For example: `LORA_FLICKER_HUB_REPO = "Khatijaliya/flicker8k-lora"`

#### **Step 4: Authenticate and Deploy**

Finally, run the following Python code in your Colab notebook to log in and deploy your project. This is how you grant the Python script access to your Hugging Face account using the **API token**.

```python
from huggingface_hub import notebook_login
from huggingface_hub import create_repo, upload_folder

# Login to Hugging Face (you will be prompted to enter your token)
notebook_login()

# Set your Hugging Face username and space name
your_username = " "
space_name = "flicker8k-generator"
space_id = f"{your_username}/{space_name}"

# Create the Space (run this only once)
create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", private=False)

# Upload the current directory (including app.py and requirements.txt) to your Space
upload_folder(
    repo_id=space_id,
    folder_path=".",        # Current directory
    path_in_repo=".",       # Upload everything to root
    repo_type="space"
)

print(f"Your Space is live at: https://huggingface.co/spaces/{space_id}")
```

