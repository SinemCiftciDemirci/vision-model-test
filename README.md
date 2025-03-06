Vision Model Test
A Python-based project that integrates text translation and image generation using Hugging Face models. This project processes fairy tale summaries, translates them from Turkish to English, and generates illustrations using Stable Diffusion. Feel free to try different translation and vision models.

📌 Features
Automatic Translation: Converts Turkish text to English using Helsinki-NLP/opus-mt-tr-en.
Image Generation: Uses Stable Diffusion (runwayml/stable-diffusion-v1-5) to create images based on the translated summaries.
Efficient Device Management: Supports CUDA if available, otherwise falls back to CPU.
Configurable and Modular: Clean separation of concerns with dedicated modules for translation, image generation, and configuration loading.

🛠️ Setup Instructions
1️⃣ Clone the Repository

git clone https://github.com/SinemCiftciDemirci/vision-model-test.git
cd vision_model_test

2️⃣ Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate      # On Windows

3️⃣ Install Required Dependencies

pip install torch transformers diffusers

🚀 Usage

1️⃣ Run the Script

python vision_model_test.py

2️⃣ Select a Fairy Tale Summary File

The script will prompt you to select a text file containing a fairy tale summary.
The summary should be divided into Introduction, Development, and Conclusion sections.

3️⃣ Results

Translated text is saved in translate_vision_test/results.txt.
Generated images are stored in translate_vision_test/ as:
Introduction.png
Development.png
Conclusion.png


📂 Project Structure

vision_model_test/
│── config_loader.py       # Handles configuration settings and device selection (CPU/GPU)
│── image_generator.py     # Loads models for translation & image generation
│── vision_model_test.py   # Main script that processes input files and generates images
│── requirements.txt       # Dependency list (to be created)
│── translate_vision_test/ # Output directory (translated text + generated images)


📜 File Descriptions

1️⃣ vision_model_test.py

Main entry point for the project.
Prompts the user to select a fairy tale summary.
Extracts Introduction, Development, and Conclusion.
Translates the text and generates corresponding images.
Saves the results in the translate_vision_test/ directory.

2️⃣ image_generator.py

Loads and applies models for:
Text translation (Helsinki-NLP/opus-mt-tr-en)
Image generation (Stable Diffusion)
Handles errors gracefully.
Uses Hugging Face Transformers and Diffusers.

3️⃣ config_loader.py

Determines the device to use (CUDA if available, otherwise CPU).
Loads any configuration files.
Prevents CUDA-related crashes by handling GPU availability checks.

🔧 Troubleshooting

1️⃣ Model Download Issues
If the models fail to download, manually clear the Hugging Face cache:

rm -rf ~/.cache/huggingface

Or for Windows:

rmdir /s /q C:\Users\YourUser\.cache\huggingface

Then restart the script.

2️⃣ CUDA Not Working
If you encounter CUDA errors, force the script to use CPU:

# Edit config_loader.py
device = torch.device("cpu")

3️⃣ No Text Extracted from the File

Ensure your text file is formatted correctly with:


Giriş:
<text>

Gelişme:
<text>

Sonuç:
<text>


or

Introduction:
<text>

Development:
<text>

Conclusion:
<text>


📜 License
This project is licensed under the MIT License.