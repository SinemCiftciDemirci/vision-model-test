import os
from tkinter import Tk, filedialog
from image_generator import (
    get_translation_models,
    get_image_pipeline,
    get_clip_tokenizer,
    translate_text
)
from config_loader import device

# Load models dynamically
translation_model, translation_tokenizer = get_translation_models()
pipe = get_image_pipeline()
clip_tokenizer = get_clip_tokenizer()

if not translation_model or not translation_tokenizer:
    print("Translation model failed to load. Exiting.")
    exit()

if not pipe:
    print("Image generation model failed to load. Exiting.")
    exit()

if not clip_tokenizer:
    print("CLIP Tokenizer failed to load. Exiting.")
    exit()

# Ask the user to select a file
def select_file():
    Tk().withdraw()  # Hide Tkinter GUI
    file_path = filedialog.askopenfilename(
        title="Select a TXT file",
        filetypes=[("Text Files", "*.txt")]
    )
    return file_path

# Process file and extract sections dynamically
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    section_map = {
        "Giriş": "Introduction",
        "Gelişme": "Development",
        "Sonuç": "Conclusion",
        "Introduction": "Introduction",
        "Development": "Development",
        "Conclusion": "Conclusion"
    }

    sections = {"Introduction": None, "Development": None, "Conclusion": None}

    for turkish_name, english_name in section_map.items():
        start_idx = content.find(f"{turkish_name}:")
        if start_idx != -1:
            end_idx = content.find("\n\n", start_idx)
            end_idx = end_idx if end_idx != -1 else len(content)
            sections[english_name] = content[start_idx + len(turkish_name) + 1:end_idx].strip()

    return sections

# Generate an image while considering CLIP prompt limits
def generate_image_with_prompt(english_summary, base_prompt, pipe, output_folder, title):
    try:
        base_tokens = clip_tokenizer.tokenize(base_prompt)
        max_total_length = 77
        max_prompt_length = max_total_length - len(base_tokens) - 2

        tokens = clip_tokenizer.tokenize(english_summary)
        if len(tokens) > max_prompt_length:
            tokens = tokens[:max_prompt_length]
            english_summary = clip_tokenizer.convert_tokens_to_string(tokens)

        prompt = f"{base_prompt} {english_summary}"
        
        image = pipe(prompt).images[0]
        os.makedirs(output_folder, exist_ok=True)
        image_path = os.path.join(output_folder, f"{title}.png")
        image.save(image_path)
        return image_path

    except Exception as e:
        return f"Image generation failed: {e}"

# Save output
def save_results(output_folder, translations, image_paths):
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "results.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for section, translated_text in translations.items():
            f.write(f"{section}:\n{translated_text}\n")
            f.write(f"Image: {image_paths.get(section, 'Image generation failed.')}\n\n")

# Main process
def main():
    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return

    sections = process_file(file_path)

    output_folder = "translate_vision_test"
    base_prompt = "A watercolor illustration of a children's story"
    translations = {}
    image_paths = {}

    for section, text in sections.items():
        if text:
            translated_text = translate_text(text, translation_model, translation_tokenizer)
            translations[section] = translated_text
            image_path = generate_image_with_prompt(translated_text, base_prompt, pipe, output_folder, section)
            image_paths[section] = image_path

    save_results(output_folder, translations, image_paths)

if __name__ == "__main__":
    main()
