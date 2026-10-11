import os
import urllib.request
import zipfile
import shutil

def download_and_extract_dataset(base_dir):
    dataset_dir = os.path.join(base_dir, "dataset")
    with_mask_dir = os.path.join(dataset_dir, "with_mask")
    without_mask_dir = os.path.join(dataset_dir, "without_mask")
    
    os.makedirs(with_mask_dir, exist_ok=True)
    os.makedirs(without_mask_dir, exist_ok=True)
    
    print("Downloading Real Face Mask Dataset from repository archive...")
    zip_url = "https://github.com/prajnasb/observations/archive/refs/heads/master.zip"
    zip_path = os.path.join(base_dir, "dataset.zip")
    
    try:
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Download finished. Extracting files...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
            
        # Move extracted files to dataset/with_mask and dataset/without_mask
        extracted_base = os.path.join(base_dir, "observations-master", "experiements", "dest_folder")
        
        # Source train and test folders
        subfolders = ["train", "test"]
        for sub in subfolders:
            wm_src = os.path.join(extracted_base, sub, "with_mask")
            wom_src = os.path.join(extracted_base, sub, "without_mask")
            
            if os.path.exists(wm_src):
                for f in os.listdir(wm_src):
                    src_f = os.path.join(wm_src, f)
                    dst_f = os.path.join(with_mask_dir, f"{sub}_{f}")
                    if os.path.isfile(src_f):
                        shutil.copy(src_f, dst_f)
                        
            if os.path.exists(wom_src):
                for f in os.listdir(wom_src):
                    src_f = os.path.join(wom_src, f)
                    dst_f = os.path.join(without_mask_dir, f"{sub}_{f}")
                    if os.path.isfile(src_f):
                        shutil.copy(src_f, dst_f)
                        
        print(f"Dataset successfully created with {len(os.listdir(with_mask_dir))} 'with_mask' images and {len(os.listdir(without_mask_dir))} 'without_mask' images!")
        
        # Clean up zip and temp folders
        if os.path.exists(zip_path):
            os.remove(zip_path)
        temp_extracted = os.path.join(base_dir, "observations-master")
        if os.path.exists(temp_extracted):
            shutil.rmtree(temp_extracted)
            
    except Exception as e:
        print(f"Error downloading/extracting dataset: {e}")
        # Fallback to direct raw image URLs if zip fails
        print("Falling back to downloading individual sample images...")
        download_samples_fallback(with_mask_dir, without_mask_dir)

def download_samples_fallback(with_mask_dir, without_mask_dir):
    base_url = "https://raw.githubusercontent.com/prajnasb/observations/master/experiements/dest_folder"
    for i in range(1, 101):
        try:
            wm_url = f"{base_url}/train/with_mask/{i}-with-mask.jpg"
            wom_url = f"{base_url}/train/without_mask/{i}.jpg"
            urllib.request.urlretrieve(wm_url, os.path.join(with_mask_dir, f"mask_{i:03d}.jpg"))
            urllib.request.urlretrieve(wom_url, os.path.join(without_mask_dir, f"no_mask_{i:03d}.jpg"))
        except Exception:
            pass
    print("Fallback download complete.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    download_and_extract_dataset(base_dir)
