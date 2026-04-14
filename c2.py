import cv2
import os

# The Universal Canvas (300 DPI US Letter)
TARGET_WIDTH = 2550
TARGET_HEIGHT = 3300

def create_master_template(input_path, output_path):
    """
    Resizes a raw blank form template to the exact mathematical grid 
    required for the extraction pipeline.
    """
    if not os.path.exists(input_path):
        print(f"❌ ERROR: Could not find input file at {input_path}")
        return

    print(f"Loading raw template: {input_path}")
    raw_img = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if raw_img is None:
        print(f"❌ ERROR: OpenCV could not read {input_path}. Check if the file is corrupted.")
        return

    # Log original size
    orig_h, orig_w = raw_img.shape[:2]
    print(f"   -> Original Size: {orig_w}w x {orig_h}h")

    # Force resize to the universal grid
    print(f"   -> Resizing to {TARGET_WIDTH}w x {TARGET_HEIGHT}h using INTER_CUBIC...")
    master_img = cv2.resize(
        raw_img, 
        (TARGET_WIDTH, TARGET_HEIGHT), 
        interpolation=cv2.INTER_CUBIC
    )

    # Save the new master template
    cv2.imwrite(output_path, master_img)
    print(f"✅ SUCCESS: Master template saved to {output_path}\n")

if __name__ == "__main__":
    # --- 1. UPDATE THESE PATHS TO WHERE YOUR RAW DOWNLOADED IMAGES ARE ---
    
    # CMS-1500 Paths
    raw_1500_path = "raw_blank_cms1500.jpg"       # Put your downloaded 1500 image here
    final_1500_path = "sample_cms1500.jpg"        # This is the output file you will use
    
    # CMS-1450 (UB-04) Paths
    raw_1450_path = "raw_blank_cms1450.jpg"       # Put your downloaded 1450 image here
    final_1450_path = "sample_cms1450.jpg"        # This is the output file you will use

    # --- 2. RUN THE RESIZER ---
    create_master_template(raw_1500_path, final_1500_path)
    create_master_template(raw_1450_path, final_1450_path)
