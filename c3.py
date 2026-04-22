def get_form_bounding_box(image):
    """Finds the absolute outer square of the form to act as cutting lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find the grid lines
    line_length = max(image.shape[1], image.shape[0]) // 40
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    
    # Merge into a thick skeleton
    grid_mask = cv2.addWeighted(v_lines, 0.5, h_lines, 0.5, 0.0)
    grid_mask = cv2.dilate(grid_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=3)
    
    # Find the largest block on the page (The Form)
    cnts, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("Could not detect the grid lines.")
        
    largest_contour = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a 10-pixel padding so we don't accidentally cut off the outer ink
    pad = 10
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = min(image.shape[1] - x, w + (pad*2)), min(image.shape[0] - y, h + (pad*2))
    
    return x, y, w, h

def preprocess_and_align(target_path, template_path):
    # 1. Load and deskew
    pil_img = Image.open(target_path)
    deskewed_cv_img = deskew_image(pil_img)

    # 2. Find the border
    try:
        x, y, w, h = get_form_bounding_box(deskewed_cv_img)
        
        if DEBUG_MODE:
            debug_canvas = deskewed_cv_img.copy()
            cv2.rectangle(debug_canvas, (x, y), (x+w, y+h), (0, 255, 0), 10)
            show_debug_image("DEBUG: Scissors Cut Line", debug_canvas)
            
        # 3. Physically crop the form out of the image
        cropped_form = deskewed_cv_img[y:y+h, x:x+w]
        
    except Exception as e:
        print(f"Crop failed: {e}. Falling back to full image.")
        cropped_form = deskewed_cv_img

    # 4. BRUTE FORCE STRETCH AND SQUEEZE
    # We forcefully resize the cropped box into the exact MS Paint dimensions
    aligned_img = cv2.resize(cropped_form, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    
    # Show the final ghost overlay
    if DEBUG_MODE:
        template_img_raw = cv2.imread(template_path, cv2.IMREAD_COLOR)
        template_img = standardize_image_size(template_img_raw)
        show_alignment_overlay(aligned_img, template_img)
        
    return aligned_img
