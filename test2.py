import os
import cv2
import json
import time
import shutil
import traceback
import pandas as pd
import numpy as np
import openpyxl  # Explicitly imported for Excel writing
from PIL import Image
from datetime import datetime
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=API_KEY)

# Gemini 2.5 Pro Pricing (Adjust as needed)
PRICE_INPUT = 7.00
PRICE_OUTPUT = 21.00

# Databricks Paths
LOCAL_WORKSPACE_DIR = "/tmp/medical_claims_processing/"
VOLUME_DIR = "/Volumes/your_catalog/your_schema/your_volume/"

INPUT_FOLDER = os.path.join(VOLUME_DIR, "input_images")
SAMPLE_CMS1500 = os.path.join(VOLUME_DIR, "templates", "sample_cms1500.jpg")
SAMPLE_CMS1450 = os.path.join(VOLUME_DIR, "templates", "sample_cms1450.jpg")

LOCAL_EXCEL = os.path.join(LOCAL_WORKSPACE_DIR, "output_data.xlsx")
VOLUME_EXCEL = os.path.join(VOLUME_DIR, "output_data.xlsx")

LOCAL_INVENTORY = os.path.join(LOCAL_WORKSPACE_DIR, "inventory.xlsx")
VOLUME_INVENTORY = os.path.join(VOLUME_DIR, "inventory.xlsx")

LOCAL_TOKENS = os.path.join(LOCAL_WORKSPACE_DIR, "token_calculation.xlsx")
VOLUME_TOKENS = os.path.join(VOLUME_DIR, "token_calculation.xlsx")

PROCESS_LOG_PATH = os.path.join(VOLUME_DIR, "process_log.txt")

os.makedirs(LOCAL_WORKSPACE_DIR, exist_ok=True)

# ==========================================
# 2. LOGGING & VOLUME SYNC UTILITIES
# ==========================================
def log_process_status(message):
    """Simple text logger matching your previous script."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(PROCESS_LOG_PATH, "a") as f:
        f.write(log_entry + "\n")

def sync_to_volume(local_path, volume_path):
    """Safely copies files to Databricks volume, removing the old one to prevent locking."""
    try:
        if os.path.exists(volume_path):
            os.remove(volume_path) 
        if os.path.exists(local_path):
            shutil.copy2(local_path, volume_path) 
    except Exception as e:
        log_process_status(f"ERROR syncing {local_path} to Volume: {str(e)}")

def load_or_create_excel(local_path, volume_path, columns):
    if os.path.exists(volume_path):
        shutil.copy2(volume_path, local_path)
        return pd.read_excel(local_path, engine='openpyxl')
    else:
        df = pd.DataFrame(columns=columns)
        df.to_excel(local_path, index=False, engine='openpyxl')
        sync_to_volume(local_path, volume_path)
        return df

def get_or_create_inventory():
    """LOCKED INVENTORY LOGIC: Sorts alphabetically to guarantee identical indices."""
    if os.path.exists(VOLUME_INVENTORY):
        log_process_status("--- LOCKED: Found existing 'inventory.xlsx'. Loading Master List... ---")
        shutil.copy2(VOLUME_INVENTORY, LOCAL_INVENTORY)
        return pd.read_excel(LOCAL_INVENTORY, engine='openpyxl')
    else:
        log_process_status(f"--- INITIALIZING: Creating NEW 'inventory.xlsx' from {INPUT_FOLDER} ---")
        valid_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
        files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_exts)]
        files.sort() # Deterministic sorting

        inventory_data = []
        for idx, f in enumerate(files):
            inventory_data.append({
                "Index": idx, 
                "File_Name": f, 
                "Status": "PENDING"
            })
        
        df = pd.DataFrame(inventory_data)
        df.to_excel(LOCAL_INVENTORY, index=False, engine='openpyxl')
        sync_to_volume(LOCAL_INVENTORY, VOLUME_INVENTORY)
        log_process_status(f"--- Inventory created with {len(df)} files. ---")
        return df

def log_token_usage_excel(filename, in_tokens, out_tokens, status="SUCCESS"):
    """Adapted cost logger with Volume syncing."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ip_cost = (in_tokens / 1_000_000) * PRICE_INPUT
    op_cost = (out_tokens / 1_000_000) * PRICE_OUTPUT
    total_cost = ip_cost + op_cost

    df = load_or_create_excel(LOCAL_TOKENS, VOLUME_TOKENS, [
        'filename', 'input_tokens', 'output_tokens', 'Total_Cost', 'processed_flag', 'processing_date'
    ])

    mask = df['filename'] == filename
    if mask.any():
        df.loc[mask, 'input_tokens'] = in_tokens
        df.loc[mask, 'output_tokens'] = out_tokens
        df.loc[mask, 'Total_Cost'] = round(total_cost, 6)
        df.loc[mask, 'processed_flag'] = status
        df.loc[mask, 'processing_date'] = current_time 
    else:
        new_row = pd.DataFrame([{
            'filename': filename,
            'input_tokens': in_tokens,
            'output_tokens': out_tokens,
            'Total_Cost': round(total_cost, 6),
            'processed_flag': status,
            'processing_date': current_time 
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_excel(LOCAL_TOKENS, index=False, engine='openpyxl')
    sync_to_volume(LOCAL_TOKENS, VOLUME_TOKENS)

# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================
def deskew_image(pil_image):
    """ROBUST DESKEW: Handles Major (90/270) and Minor (±5) rotations."""
    if pil_image is None: return None
    print("\n   --- [Deskew] Starting Analysis ---")
    
    img = np.array(pil_image)
    if len(img.shape) == 3:
        if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    aspect_ratios = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20: 
            aspect_ratios.append(h / w)
            
    median_ratio = np.median(aspect_ratios) if aspect_ratios else 0.5 
    rotation_needed = 0
    
    if median_ratio > 1.2:
        print(f"   -> Detected VERTICAL text (Ratio: {median_ratio:.2f}). Rotating 90 deg.")
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        thresh = cv2.rotate(thresh, cv2.ROTATE_90_CLOCKWISE) 
        rotation_needed = 90

    print("   -> Calculating optimal projection angle...")
    h, w = img.shape[:2]
    scores = []
    angles = np.arange(-5, 5.1, 0.5) 
    
    for angle in angles:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated_thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST)
        hist = np.sum(rotated_thresh, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        scores.append(score)

    best_micro_angle = angles[np.argmax(scores)]
    print(f"   -> Best Fine-Tune Angle: {best_micro_angle:.2f} degrees")
    
    if rotation_needed != 0 or abs(best_micro_angle) > 0.1:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_micro_angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(255, 255, 255)
        )
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def upscale_and_denoise(image):
    """Upscales low-res images and applies sharpening for better LLM Vision extraction."""
    print("   -> [Enhance] Upscaling and applying clarity filters...")
    
    # 1. Cubic Upscale
    height, width = image.shape[:2]
    if width < MIN_IMAGE_WIDTH:
        scale_factor = MIN_IMAGE_WIDTH / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
    # 2. CLAHE Contrast Normalization (Reduces scanner shadows/noise)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 3. Edge Sharpening Kernel (Crisps up blurry text)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    enhanced = cv2.filter2D(image, -1, kernel)
    
    return enhanced

def preprocess_image(image_path):
    pil_img = Image.open(image_path)
    pil_img = deskew_image(pil_img)
    
    open_cv_image = np.array(pil_img)
    if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
        
    cv_img = crop_and_resize(open_cv_image)
    cv_img = upscale_and_denoise(cv_img)
    
    temp_path = os.path.join(LOCAL_WORKSPACE_DIR, "temp_processed.jpg")
    cv2.imwrite(temp_path, cv_img)
    return temp_path
# ==========================================
# 4. LLM PROMPT & EXTRACTION (NEW SDK)
# ==========================================
def get_unified_prompt():
    return """
    You are an expert medical billing data extractor. I have provided:
    1. A reference sample of a blank CMS-1500 form.
    2. A reference sample of a blank CMS-1450 (UB-04) form.
    3. The Target Image (filled form).

    TASK 1: Determine if the Target Image is a CMS-1500 or CMS-1450.
    TASK 2: Extract data into the JSON schema below. Use null if a field is blank.

    CRITICAL ACCURACY RULES FOR NUMBERS & TEXT:
    - EXACT TRANSCRIPTION: Extract data character-for-character. Do NOT auto-correct, guess, or reformat values.
    - PRESERVE NUMBERS AS STRINGS: Do NOT drop leading or trailing zeros (e.g., "00456" must remain "00456").
    - IGNORE FORM LINES: Ignore the printed red or black box lines that may intersect with the typed text. 

    CMS-1500 SPECIFIC RULES:
    - Block 1 (Line of Business): There are 7 checkboxes at the top. Return the text label of the box that is marked (e.g., "MEDICARE", "MEDICAID", "TRICARE", "CHAMPVA", "GROUP HEALTH PLAN", "FECA BLK LUNG", "OTHER").
    - YES/NO Checkboxes (Blocks 10a, 10b, 10c, and 27): These contain separate "YES" and "NO" boxes. Return the string "YES" if the Yes box is marked, "NO" if the No box is marked, or null if neither is marked.
    - Zip Codes (Blocks 32, 33): Isolate the zip code from the address string.
    - Block 21: Extract all listed diagnoses as a single comma-separated string.
    - Block 24 (Table): There are 6 distinct rows. Extract the data for EACH cell into its specific designated JSON key. If a row is entirely blank, return null for all fields in that row.

    Return ONLY raw JSON. No markdown formatting like ```json.
    {
      "Form_Type": "CMS-1500",
      "CMS_1500_Data": {
        "1_LOB": "", "1a_Insured_ID": "", "2_Patient_Name": "", "3_DOB": "", "3_Gender": "",
        "4_Insured_Name": "", "5_Patient_Address": "", "5_Patient_City": "", "5_Patient_State": "",
        "5_Patient_Zip": "", "5_Patient_Phone": "", "6_Relationship": "", "7_Insured_Address": "",
        "7_Insured_City": "", "7_Insured_State": "", "7_Insured_Zip": "", "7_Insured_Phone": "",
        "8_Reserved_1": "", "9_Other_Insured_Name": "", "9a_Other_Policy_Num": "", "9b_Reserved_2": "",
        "9c_Reserved_3": "", "9d_Insurance_Plan_Name": "", 
        "10a_Employment": "", "10b_Auto_Accident": "", "10c_Other_Accident": "", "10d_Claim_Codes": "",
        "11_FECA_Number": "", "11a_Insured_DOB": "", "11a_Gender": "", "11b_Other_Claim_ID": "",
        "11c_Insurance_Plan_Name": "", "11d_Other_Health_Plan": "", "12_Patient_Sign_IND": "",
        "12_Date_of_Signing": "", "13_Insured_Sign_IND": "", "14_Date_Current_Condition": "",
        "15_Other_Date": "", "16_From_Date": "", "16_To_Date": "", "17_Referring_Provider_Name": "",
        "17a_Provider_ID": "", "17b_NPI": "", "18_Hospitalization_From_Date": "", "18_Hospitalization_To_Date": "",
        "19_Additional_Info": "", "20_Outside_Lab": "", "20_Charges": "", "21_Diagnosis": "",
        "21_ICD_Ind": "", "22_Resubmission_Code": "", "22_Original_Ref_No": "", "23_Prior_Auth_Number": "",
        
        "24_1A_Service_From_Date": "", "24_1A_Service_To_Date": "", "24_1B_Place_Of_Service": "", "24_1C_EMG": "", "24_1D_CPT_HCPCS": "", "24_1D_Modifier": "", "24_1E_Diagnosis_Pointer": "", "24_1F_Charges": "", "24_1G_Units": "", "24_1H_Family_Plan": "", "24_1I_NPI_Indicator": "", "24_1J_Provider_ID": "", "24_1J_NPI": "",
        "24_2A_Service_From_Date": "", "24_2A_Service_To_Date": "", "24_2B_Place_Of_Service": "", "24_2C_EMG": "", "24_2D_CPT_HCPCS": "", "24_2D_Modifier": "", "24_2E_Diagnosis_Pointer": "", "24_2F_Charges": "", "24_2G_Units": "", "24_2H_Family_Plan": "", "24_2I_NPI_Indicator": "", "24_2J_Provider_ID": "", "24_2J_NPI": "",
        "24_3A_Service_From_Date": "", "24_3A_Service_To_Date": "", "24_3B_Place_Of_Service": "", "24_3C_EMG": "", "24_3D_CPT_HCPCS": "", "24_3D_Modifier": "", "24_3E_Diagnosis_Pointer": "", "24_3F_Charges": "", "24_3G_Units": "", "24_3H_Family_Plan": "", "24_3I_NPI_Indicator": "", "24_3J_Provider_ID": "", "24_3J_NPI": "",
        "24_4A_Service_From_Date": "", "24_4A_Service_To_Date": "", "24_4B_Place_Of_Service": "", "24_4C_EMG": "", "24_4D_CPT_HCPCS": "", "24_4D_Modifier": "", "24_4E_Diagnosis_Pointer": "", "24_4F_Charges": "", "24_4G_Units": "", "24_4H_Family_Plan": "", "24_4I_NPI_Indicator": "", "24_4J_Provider_ID": "", "24_4J_NPI": "",
        "24_5A_Service_From_Date": "", "24_5A_Service_To_Date": "", "24_5B_Place_Of_Service": "", "24_5C_EMG": "", "24_5D_CPT_HCPCS": "", "24_5D_Modifier": "", "24_5E_Diagnosis_Pointer": "", "24_5F_Charges": "", "24_5G_Units": "", "24_5H_Family_Plan": "", "24_5I_NPI_Indicator": "", "24_5J_Provider_ID": "", "24_5J_NPI": "",
        "24_6A_Service_From_Date": "", "24_6A_Service_To_Date": "", "24_6B_Place_Of_Service": "", "24_6C_EMG": "", "24_6D_CPT_HCPCS": "", "24_6D_Modifier": "", "24_6E_Diagnosis_Pointer": "", "24_6F_Charges": "", "24_6G_Units": "", "24_6H_Family_Plan": "", "24_6I_NPI_Indicator": "", "24_6J_Provider_ID": "", "24_6J_NPI": "",
        
        "25_Tax_ID": "", "26_Patient_Acc_Number": "", "27_Accept_Assignment": "", "28_Total_Charges": "",
        "29_Amount_Paid": "", "30_Reserved_4": "", "31_Is_Signed": "", "31_Date_of_Signature": "",
        "32_Facility_Location": "", "32_Facility_Zip": "", "32a_NPI": "", "32b_Provider_ID": "",
        "33_Billing_Facility_Location": "", "33_Billing_Facility_Zip": "", "33a_NPI": "", "33b_Provider_ID": ""
      },
      "CMS_1450_Data": {
        "1_Billing_Provider_Name": "", "1_Billing_Provider_Address": "", "1_Billing_Provider_Phone": "",
        "2_Pay_To_Name_Address": "",
        "3a_Pat_Cntl_Num": "", "3b_Med_Rec_Num": "",
        "4_Type_of_Bill": "", "5_Fed_Tax_Num": "",
        "6_Statement_From": "", "6_Statement_Through": "",
        "8_Patient_Name": "",
        "9_Patient_Address": "", "9_Patient_City": "", "9_Patient_State": "", "9_Patient_Zip": "",
        "10_Patient_DOB": "", "11_Patient_Sex": "",
        "12_Admission_Date": "", "13_Admission_Hr": "", "14_Admission_Type": "", "15_Admission_Src": "", "17_Patient_Stat": "",
        "18_28_Condition_Codes": "",
        "31_34_Occurrence_Codes_Dates": "",
        "35_36_Occurrence_Spans": "",
        "39_41_Value_Codes_Amounts": "",
        
        "42_47_Row1_RevCode": "", "42_47_Row1_Desc": "", "42_47_Row1_HCPCS": "", "42_47_Row1_Date": "", "42_47_Row1_Units": "", "42_47_Row1_Total": "",
        "42_47_Row2_RevCode": "", "42_47_Row2_Desc": "", "42_47_Row2_HCPCS": "", "42_47_Row2_Date": "", "42_47_Row2_Units": "", "42_47_Row2_Total": "",
        "42_47_Row3_RevCode": "", "42_47_Row3_Desc": "", "42_47_Row3_HCPCS": "", "42_47_Row3_Date": "", "42_47_Row3_Units": "", "42_47_Row3_Total": "",
        "42_47_Row4_RevCode": "", "42_47_Row4_Desc": "", "42_47_Row4_HCPCS": "", "42_47_Row4_Date": "", "42_47_Row4_Units": "", "42_47_Row4_Total": "",
        "42_47_Row5_RevCode": "", "42_47_Row5_Desc": "", "42_47_Row5_HCPCS": "", "42_47_Row5_Date": "", "42_47_Row5_Units": "", "42_47_Row5_Total": "",
        "42_47_Row6_RevCode": "", "42_47_Row6_Desc": "", "42_47_Row6_HCPCS": "", "42_47_Row6_Date": "", "42_47_Row6_Units": "", "42_47_Row6_Total": "",
        "42_47_Row7_RevCode": "", "42_47_Row7_Desc": "", "42_47_Row7_HCPCS": "", "42_47_Row7_Date": "", "42_47_Row7_Units": "", "42_47_Row7_Total": "",
        "42_47_Row8_RevCode": "", "42_47_Row8_Desc": "", "42_47_Row8_HCPCS": "", "42_47_Row8_Date": "", "42_47_Row8_Units": "", "42_47_Row8_Total": "",
        "42_47_Row9_RevCode": "", "42_47_Row9_Desc": "", "42_47_Row9_HCPCS": "", "42_47_Row9_Date": "", "42_47_Row9_Units": "", "42_47_Row9_Total": "",
        "42_47_Row10_RevCode": "", "42_47_Row10_Desc": "", "42_47_Row10_HCPCS": "", "42_47_Row10_Date": "", "42_47_Row10_Units": "", "42_47_Row10_Total": "",
        "47_Total_Line_Amount": "",
        
        "50A_Payer_Name": "", "50B_Payer_Name": "", "50C_Payer_Name": "",
        "51A_Health_Plan_ID": "", "51B_Health_Plan_ID": "", "51C_Health_Plan_ID": "",
        "56A_NPI": "", "56B_NPI": "", "56C_NPI": "",
        "58A_Insured_Name": "", "58B_Insured_Name": "", "58C_Insured_Name": "",
        "59A_P_Rel": "", "59B_P_Rel": "", "59C_P_Rel": "",
        "60A_Insured_Unique_ID": "", "60B_Insured_Unique_ID": "", "60C_Insured_Unique_ID": "",
        "63A_Treatment_Auth": "", "63B_Treatment_Auth": "", "63C_Treatment_Auth": "",
        
        "66_DX_Version_Qual": "", "67_Principal_Diag_Code": "", "67_Other_Diag_Codes": "",
        "74_Principal_Procedure_Code_Date": "", "74_Other_Procedure_Codes_Dates": "",
        "76_Attending_Provider_NPI": "", "76_Attending_Provider_Name": "",
        "80_Remarks": ""
      }
    }
    """

def create_image_part(image_path):
    """Helper to read image bytes and format them for the new SDK."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Missing image file: {image_path}")
        
    with open(image_path, "rb") as img:
        image_bytes = img.read()
    
    # Using image/jpeg because our preprocessing saves 'temp_processed.jpg'
    return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

def process_image_with_llm(target_path):
    try:
        # 1. Create the text prompt part
        text_part = types.Part.from_text(text=get_unified_prompt())
        
        # 2. Read the bytes for all three images
        sample_1500_part = create_image_part(SAMPLE_CMS1500)
        sample_1450_part = create_image_part(SAMPLE_CMS1450)
        target_part = create_image_part(target_path)
        
        # 3. Assemble the contents list (Text prompt first, then images)
        contents = [text_part, sample_1500_part, sample_1450_part, target_part]
        
        # 4. Call the model (Enforcing JSON output via GenerateContentConfig)
        response = client.models.generate_content(
            model="gemini-2.5-pro", 
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0  # Zero temperature for maximum extraction determinism
            )
        )
        
        # 5. Safely extract token usage
        in_tok = 0
        out_tok = 0
        if response.usage_metadata:
            in_tok = response.usage_metadata.prompt_token_count
            out_tok = response.usage_metadata.candidates_token_count
        
        return json.loads(response.text), in_tok, out_tok
        
    except Exception as e:
        log_process_status(f"LLM API Error: {str(e)}")
        return None, 0, 0

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def process_pipeline(start_index, end_index):
    df_inventory = get_or_create_inventory()
    df_output = load_or_create_excel(LOCAL_EXCEL, VOLUME_EXCEL, ["File_Name", "Form_Type"])

    if df_inventory.empty:
        log_process_status("No files to process. Exiting.")
        return

    # Bounds check
    if start_index < 0: start_index = 0
    if end_index >= len(df_inventory): end_index = len(df_inventory) - 1

    log_process_status(f"--- Pipeline Started: Processing Index {start_index} to {end_index} ---")
    
    index = start_index

    # --- MAIN LOOP ---
    while index <= end_index:
        try:
            row = df_inventory.iloc[index]
            file_name = row['File_Name']
            
            print(f"\n[{index}] Processing: {file_name}")

            # --- A. SKIP CHECK ---
            if file_name in df_output["File_Name"].values:
                print(f"   -> Output found in Excel. Skipping extraction.")
                index += 1
                continue

            # --- B. PROCESS FILE ---
            file_path = os.path.join(INPUT_FOLDER, file_name)
            
            if not os.path.exists(file_path):
                log_process_status(f"   -> ERROR: File not found: {file_path}")
                index += 1
                continue

            processed_path = preprocess_image(file_path)
            if not processed_path: 
                raise Exception("Preprocessing failed.")
            
            extracted_json, in_tok, out_tok = process_image_with_llm(processed_path)
            if not extracted_json: 
                raise Exception("Failed to extract JSON from LLM.")
            
            # --- C. PARSE DATA ---
            form_type = extracted_json.get("Form_Type", "UNKNOWN")
            row_data = {"File_Name": file_name, "Form_Type": form_type}
            
            if form_type == "CMS-1500":
                row_data.update(extracted_json.get("CMS_1500_Data", {}))
            elif form_type == "CMS-1450":
                row_data.update(extracted_json.get("CMS_1450_Data", {}))
                
            df_new_row = pd.DataFrame([row_data])
            df_output = pd.concat([df_output, df_new_row], ignore_index=True)
            
            # Save Output immediately
            df_output.to_excel(LOCAL_EXCEL, index=False, engine='openpyxl')
            sync_to_volume(LOCAL_EXCEL, VOLUME_EXCEL)
            
            # Update Inventory Status
            df_inventory.loc[index, "Status"] = "SUCCESS"
            df_inventory.to_excel(LOCAL_INVENTORY, index=False, engine='openpyxl')
            sync_to_volume(LOCAL_INVENTORY, VOLUME_INVENTORY)

            # Log Cost
            log_token_usage_excel(file_name, in_tok, out_tok, status="SUCCESS")
            
            log_process_status(f"Finished {file_name}")

            if index < end_index:
                print("   -> Waiting 30 seconds before proceeding to next file...")
                time.sleep(30)
            index += 1

        except Exception as e:
            print(f"!!! ERROR on Index {index}: {e}")
            log_process_status(f"ERROR: Index {index} - {traceback.format_exc()}")
            log_token_usage_excel(file_name, 0, 0, status="FAILED") # Log failure
            index += 1
            time.sleep(1) # Brief pause to stabilize before next file
            continue

if __name__ == "__main__":
    # --- DATABRICKS WIDGETS ---
    try:
        dbutils.widgets.text("start_index", "0", "Start Index")
        dbutils.widgets.text("end_index", "100", "End Index")
    except:
        pass 

    s_val = dbutils.widgets.get("start_index")
    e_val = dbutils.widgets.get("end_index")
    
    start = int(s_val) if s_val.strip() else 0
    end = int(e_val) if e_val.strip() else 999999

    process_pipeline(start_index=start, end_index=end)






import fitz  # PyMuPDF
import os

def pdf_to_images(pdf_path, output_dir, dpi=500):
    """
    Converts a PDF to high-resolution PNG images using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory to save the output images.
        dpi (int): Target DPI for the images. 500 is excellent for OCR.
        
    Returns:
        list: A list of file paths to the generated images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    saved_image_paths = []

    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        
        # PyMuPDF renders at 72 DPI by default. We calculate a zoom factor to reach the target DPI.
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        print(f"Converting '{base_name}.pdf' ({len(doc)} pages) at {dpi} DPI...")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Render the page to a pixmap (image)
            # alpha=False ensures a white background instead of transparent
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Define output path and save
            output_filename = f"{base_name}_page_{page_num + 1}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            pix.save(output_path)
            saved_image_paths.append(output_path)
            
        doc.close()
        print("Conversion complete.")
        return saved_image_paths

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []

# --- Example Usage ---
# input_pdf = "/Volumes/your_catalog/your_schema/your_volume/input_files/sample_claim.pdf"
# output_folder = "/tmp/medical_claims_processing/converted_images/"
# generated_images = pdf_to_images(input_pdf, output_folder, dpi=500)
