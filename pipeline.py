import os
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import time
import argparse

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import gc

    
def parse_args():
    parser = argparse.ArgumentParser(description="Generate a dataset of images with people inserted using the FLUX.2-Klein model.")
    parser.add_argument("--dataset_csv", type=str, default=None)
    return parser.parse_args()


def get_mask(bg_image, generated_image, threshold=60):
    diff = cv2.absdiff(cv2.cvtColor(np.array(bg_image), cv2.COLOR_RGB2GRAY), cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2GRAY))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((31, 31), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(mask)

device = "cuda"
dtype = torch.bfloat16

args = parse_args()


ground_truth_df = pd.read_csv("balanced_dataset/filtered_images.csv")
faces_df = pd.read_csv("faces/selected_faces.csv")

# Create folders for output if they don't exist
os.makedirs("dataset", exist_ok=True)
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)
dataset_df = pd.DataFrame(columns=["image_path", "lat", "lon", "source_dataset", "scene_type", "attribute", "race", "gender", "age", "mask_path", "id_face", "face_path", "gt_path", "clothing_type", "prompt", "abnormal"])
if args.dataset_csv is not None:
    dataset_df_path = args.dataset_csv
else:
    if os.path.exists("dataset/dataset_metadata.csv"):
        dataset_df_path = "dataset/dataset_metadata" + str(int(time.time())) + ".csv"  # To avoid overwriting existing metadata
    else:
        dataset_df_path = "dataset/dataset_metadata.csv"
    dataset_df.to_csv(dataset_df_path, index=False)

races = faces_df["race"].unique().tolist()
print(f"The considered races are: {races}")


cultural_attire = {
    "White": {
        "masculine": ["suit and tie", "jeans and t-shirt", "chinos with polo shirt", "bomber jacket", "cable-knit sweater", "trench coat", "flannel shirt with jeans", "tuxedo"],
        "feminine": ["sundress", "jeans and blouse", "pencil skirt with top", "trench coat", "wrap dress", "cardigan with slacks", "maxi skirt with blouse", "cocktail dress"]
    },
    "Black": {
        "masculine": ["dashiki", "suit and tie", "streetwear hoodie and joggers", "agbada", "senator suit", "boubou", "kente cloth shirt", "kaftan"],
        "feminine": ["ankara wrap dress", "dashiki blouse with trousers", "kente cloth dress", "gele headwrap with boubou", "iro and buba", "lappa skirt with top", "kitenge dress", "adire wrap dress"]
    },
    "Southeast Asian": {
        "masculine": ["batik shirt", "sarong with shirt", "barong tagalog", "longyi", "baju melayu", "pha nung", "puso ng saging", "teluk belanga"],
        "feminine": ["kebaya", "ao dai", "sarong with blouse", "baro't saya", "baju kurung", "pha sin wrap skirt", "inabel dress", "nyonya kebaya"]
    },
    "East Asian": {
        "masculine": ["changshan (tang suit)", "hanbok (men's jeogori and baji)", "hakama with kimono", "modern suit", "mandarin collar jacket", "haori", "magua jacket", "durumagi overcoat"],
        "feminine": ["qipao (cheongsam)", "hanbok (women's jeogori and chima)", "kimono", "hanfu", "furisode", "jeogori with maxi chima", "ruqun", "uchikake"]
    },
    "Latino_Hispanic": {
        "masculine": ["guayabera shirt", "charro suit", "poncho with trousers", "linen shirt and slacks", "ruana cloak", "jorongo poncho", "montuno shirt", "bombacha trousers"],
        "feminine": ["huipil", "pollera", "china poblana", "floral wrap skirt with blouse", "mestiza dress", "cholita layered skirt", "jorongo shawl with dress", "rebozo wrap with blouse"]
    },
    "Middle Eastern": {
        "masculine": ["thobe (dishdasha)", "keffiyeh with agal", "shalwar kameez", "jalabiya", "bisht cloak", "sirwal trousers with tunic", "jubba", "egal with ghutra"],
        "feminine": ["abaya", "kaftan", "jilbab with hijab", "thobe al-nashal", "farasha dress", "jalabiya with tarha", "bisht evening cloak", "shayla with maxi dress"]
    },
    "Indian": {
        "masculine": ["kurta with churidar", "sherwani", "dhoti with angavastram", "nehru jacket", "achkan coat", "bandhgala suit", "lungi with shirt", "pathani suit"],
        "feminine": ["saree", "salwar kameez", "lehenga choli", "anarkali suit", "churidar with kurti", "gharara", "sharara with dupatta", "ikat silk dress"]
    }
}

# Loop through the background images
total_images = len(ground_truth_df) * 7 * 2 * 2 # 7 races x m/f x typical clothes / non-typical clothes
rows_list = []
checkpoint_interval = 10
pipe = None
if not os.path.exists(dataset_df_path):
    pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9b-kv",
    torch_dtype=dtype,
    )
    pipe.set_progress_bar_config(disable=True)  # Disable the default progress bar to avoid cluttering the output
    pipe.to(device)
    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="max-autotune",
        fullgraph=True,
        dynamic=True,
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode,
        mode="max-autotune",
        fullgraph=True,
        dynamic=True,
    )
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()
    
    with tqdm(total=total_images, desc="Generating images") as pbar:
        for idx, row in ground_truth_df.iterrows():
            bg_image_path = row["dataset_path"]
            base_filename = os.path.basename(bg_image_path).split(".")[0]
            lat = row["lat"]
            lon = row["lon"]
            source_dataset = row["source_dataset"]
            scene_type = row["scene_type"]
            attribute = row["attribute"]
            ref_image_1 = load_image(bg_image_path)
            new_width = (ref_image_1.width // 16) * 16
            new_height = (ref_image_1.height // 16) * 16
            ref_image_1 = ref_image_1.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Loop through the races
            for race in races:
                for gender in ["Male", "Female"]:
                    face = faces_df[(faces_df["race"] == race) & (faces_df["gender"] == gender)].sample(1)
                    id_face = face["original_id"].values[0]
                    face_image_path = os.path.join("faces", face["filename"].values[0])
                    age = face["age"].values[0]
                    face_image = load_image(face_image_path)
                    id_face = int(id_face)  # Ensure it's an integer for seeding
                    prompt = f"""
                        Generate a {race.lower()} {gender.lower()} person of age {age} years in image 1 looking like the person from image 2.
                        Make the generated person look as similar as possible to the person in image 2. Make the person pretty.
                        Keep the lighting and the background as in image 1.
                        (Make the generated person face the camera. : 0.5).
                        Give the picture a natural feeling, as if it was taken by a camera. (Make the generated person look like they belong in the scene: 1.4).
                        Keep a natural pose.
                        (Do not make more than two arms, legs or faces)
                        """
                    result = pipe(
                        prompt=prompt,
                        image=[ref_image_1, face_image],   # <-- multi-reference list here
                        height=ref_image_1.height,
                        width=ref_image_1.width,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=torch.Generator(device=device).manual_seed(id_face),
                    ).images[0]


                    save_path = f"dataset/images/{base_filename}_{race.lower()}_{gender.lower()}_00.jpg"
                    result.save(save_path)
                    mask = get_mask(ref_image_1, result)
                    mask_path = f"dataset/masks/{base_filename}_{race.lower()}_{gender.lower()}_00_mask.png"
                    mask.save(mask_path)
                    rows_list.append({
                        "image_path": save_path,
                        "lat": lat,
                        "lon": lon,
                        "source_dataset": source_dataset,
                        "scene_type": scene_type,
                        "attribute": attribute,
                        "race" : race,
                        "gender": gender,
                        "age": age,
                        "mask_path": mask_path,
                        "id_face": id_face,
                        "face_path": face_image_path,
                        "gt_path": bg_image_path,
                        "clothing_type": None,
                        "prompt": prompt,
                        "abnormal": None
                    })
                    pbar.update(1)

                    # Generate the paired variant with attire associated with the selected group.
                    clothing_gender = "masculine" if gender == "Male" else "feminine"
                    clothing_options = cultural_attire[race][clothing_gender]
                    clothing_option = clothing_options[id_face % len(clothing_options)]  # Just to have a deterministic choice based on the face ID
                    prompt = f"""
                        Generate a {race.lower()} {gender.lower()} person of age {age} years in image 1 looking like the person from image 2.
                        Make the generated person look as similar as possible to the person in image 2. Make the person pretty.
                        (The person is wearing typical {clothing_gender} {clothing_option}:1.4)
                        Keep the lighting and the background as in image 1.
                        (Make the generated person face the camera. : 0.5).
                        Give the picture a natural feeling, as if it was taken by a camera. (Make the generated person look like they belong in the scene: 1.4).
                        Keep a natural pose.
                        (Do not make more than two arms, legs or faces)
                        """
                    result = pipe(
                        prompt=prompt,
                        image=[ref_image_1, face_image],   # <-- multi-reference list here
                        height=ref_image_1.height,
                        width=ref_image_1.width,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=torch.Generator(device=device).manual_seed(id_face),
                    ).images[0]


                    save_path = f"dataset/images/{base_filename}_{race.lower()}_{gender.lower()}_01.jpg"
                    result.save(save_path)
                    mask = get_mask(ref_image_1, result)
                    mask_path = f"dataset/masks/{base_filename}_{race.lower()}_{gender.lower()}_01_mask.png"
                    mask.save(mask_path)
                    rows_list.append({
                        "image_path": save_path,
                        "lat": lat,
                        "lon": lon,
                        "source_dataset": source_dataset,
                        "scene_type": scene_type,
                        "attribute": attribute,
                        "race" : race,
                        "gender": gender,
                        "age": age,
                        "mask_path": mask_path,
                        "id_face": id_face,
                        "face_path": face_image_path,
                        "gt_path": bg_image_path,
                        "clothing_type": "typical",
                        "prompt": prompt,
                        "abnormal": None
                    })
                    pbar.update(1)
            if len(rows_list) >= checkpoint_interval:
                checkpoint_interval = 100
                # Save intermediate results to avoid losing progress
                tmp_df = pd.DataFrame(rows_list)
                tmp_df.to_csv(dataset_df_path, mode="a", index=False, header=not os.path.exists(dataset_df_path))
                tqdm.write(f"Checkpoint saved with {len(rows_list)} new entries.")
                rows_list = []  # Clear the list after saving
    # Save the dataset metadata to a CSV file
    if rows_list:  # Save any remaining rows
        tmp_df = pd.DataFrame(rows_list)
        tmp_df.to_csv(dataset_df_path, mode="a", index=False, header=not os.path.exists(dataset_df_path))
dataset_df = pd.read_csv(dataset_df_path)  # Load the complete dataset metadata for the next steps
dataset_df["abnormal"] = dataset_df["abnormal"].astype(bool)
# After the loop, check if there are any abnormal images and correct them
# 0. offload the flux pipeline
tries = 0
while ((dataset_df[dataset_df['abnormal'] == True].shape[0] > 0 or tries==0) and tries < 100):
    if pipe is not None:
        del pipe
    gc.collect()
    torch.cuda.empty_cache()
    # 1. load qwen3vl 8b
    qwen_id = "Qwen/Qwen3-VL-8B-Instruct"
    # qwen_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(qwen_id, torch_dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(qwen_id)
    # 2. loop through the generated images and check for abnormalities
    for idx, row in tqdm(dataset_df.iterrows(), total=dataset_df.shape[0], desc="Checking for abnormalities"):
        image_path = row["image_path"]
        if row["abnormal"] is not False:  # Only check images that are not already marked as normal
            tqdm.write(f"Checking image: {image_path} - index: {idx}")
            text = """Is this picture strange? Answer with "Yes" if there is anything strange or abnormal in the image, such as more than 2 arms, more than 2 legs, more than 2 faces, missing body parts, or any glaring issue with the generated person. If you don't see any person in the image, or if the person is very small and hard to see, consider it an abnormality as well. Answer with "No" if there is nothing strange or abnormal in the image."""
            messages = [
                {
                    "role":"user",
                    "content": [
                        {
                            "type": "image",
                            "image" : image_path,
                        },
                        {
                            "type": "text",
                            "text" : text 
                        }
                    ]}
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs_ids = qwen.generate(**inputs, max_new_tokens=10, do_sample=False)
            generated_ids = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, outputs_ids)
            ]
            raw_answer = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # Parse yes/no
            cleaned = raw_answer.strip().lower().rstrip(".!?;,")
            if "yes" in cleaned:
                dataset_df.at[int(idx), "abnormal"] = True
            elif "no" in cleaned:
                dataset_df.at[int(idx), "abnormal"] = False
            else:
                dataset_df.at[int(idx), "abnormal"] = True  # If the answer is unclear, mark as abnormal

    dataset_df.to_csv("dataset/dataset_metadata_checked.csv", index=False)  # Save the updated dataset with abnormality labels
    print(f"Abnormality check completed. Abnormal images: {dataset_df[dataset_df['abnormal'] == True].shape[0]}. Normal images: {dataset_df[dataset_df['abnormal'] == False].shape[0]}.")
    if dataset_df[dataset_df["abnormal"] == True].shape[0] == 0:
        print("Abnormality check completed with no abnormal images remaining.")
        break
    # Put qwen to cpu to free up GPU memory
    qwen.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9b-kv",
        torch_dtype=dtype,
    )
    pipe.set_progress_bar_config(disable=True)  # Disable the default progress bar to avoid cluttering the output
    pipe.to(device)
    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="max-autotune",
        fullgraph=True,
        dynamic=True,
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode,
        mode="max-autotune",
        fullgraph=True,
        dynamic=True,
    )
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # Reload the flux pipeline for the next iteration if there are still abnormal images to check
    for idx, row in tqdm(dataset_df.iterrows(), total=dataset_df.shape[0], desc="Regenerating abnormal images @ tries " + str(tries)):
        if row["abnormal"] == True:
            filename = row["image_path"]
            mask_path = row["mask_path"]
            prompt = row["prompt"]
            gt_path = row["gt_path"]
            ref_image_1 = load_image(gt_path)
            new_width = (ref_image_1.width // 16) * 16
            new_height = (ref_image_1.height // 16) * 16
            ref_image_1 = ref_image_1.resize((new_width, new_height), Image.Resampling.LANCZOS)
            face_image = load_image(row["face_path"])
            id_face = row["id_face"]
            result = pipe(
                prompt=prompt,
                image=[ref_image_1, face_image],   # <-- multi-reference list here
                height=ref_image_1.height,
                width=ref_image_1.width,
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=torch.Generator(device=device).manual_seed(id_face + 1 + tries),
            ).images[0]
            save_path = f"{filename}"
            result.save(save_path)
            mask = get_mask(ref_image_1, result)
            mask_path = f"{mask_path}"
            mask.save(mask_path)
    tries += 1

rows_list = []
# Final thing to do:
for idx, row in tqdm(ground_truth_df.iterrows(), total=ground_truth_df.shape[0], desc="Copying ground truth images"):
    gt_image_path = row["dataset_path"]
    base_filename = os.path.basename(gt_image_path).split(".")[0]
    lat = row["lat"]
    lon = row["lon"]
    source_dataset = row["source_dataset"]
    scene_type = row["scene_type"]
    attribute = row["attribute"]
    ref_image_1 = load_image(gt_image_path)
    new_width = (ref_image_1.width // 16) * 16
    new_height = (ref_image_1.height // 16) * 16
    ref_image_1 = ref_image_1.resize((new_width, new_height), Image.Resampling.LANCZOS)
    save_path = f"dataset/images/{base_filename}_gt.jpg"
    ref_image_1.save(save_path)
    mask = torch.zeros((ref_image_1.height, ref_image_1.width), dtype=torch.uint8)
    mask = Image.fromarray(mask.numpy())
    mask_path = f"dataset/masks/{base_filename}_gt_mask.png"
    mask.save(mask_path)
    rows_list.append({
        "image_path": save_path,
        "lat": lat,
        "lon": lon,
        "source_dataset": source_dataset,
        "scene_type": scene_type,
        "attribute": attribute,
        "race" : None,
        "gender": None,
        "age": None,
        "mask_path": mask_path,
        "id_face": None,
        "face_path": None,
        "gt_path": gt_image_path,
        "clothing_type": None,
        "prompt": "ground_truth",
        "abnormal": False
    })
print(f"len(rows_list): {len(rows_list)}")
tmp_df = pd.DataFrame(rows_list)
print(tmp_df.head())
tmp_df.to_csv("dataset/dataset_metadata_checked.csv", mode="a", index=False, header=not os.path.exists("dataset/dataset_metadata_checked.csv"))
print(pd.read_csv("dataset/dataset_metadata_checked.csv").shape[0])
print("Generation pipeline completed.")





    



