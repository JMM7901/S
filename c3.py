def get_form_bounding_box(image, debug_name="Image"):
    """Finds the absolute outer square of the form grid."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find grid lines
    line_length = max(image.shape[1], image.shape[0]) // 40
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    
    # Merge and dilate to make a thick skeleton
    grid_mask = cv2.addWeighted(v_lines, 0.5, h_lines, 0.5, 0.0)
    grid_mask = cv2.dilate(grid_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=3)
    
    # Find the largest rectangular block
    cnts, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError(f"Could not detect grid lines in {debug_name}.")
        
    largest_contour = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    if DEBUG_MODE:
        debug_canvas = image.copy()
        cv2.rectangle(debug_canvas, (x, y), (x+w, y+h), (0, 255, 0), 10)
        show_debug_image(f"DEBUG: Bounding Box for {debug_name}", debug_canvas)
        
    return x, y, w, h



def preprocess_and_align(target_path, template_path):
    # 1. Load images
    pil_img = Image.open(target_path)
    deskewed_cv_img = deskew_image(pil_img)
    
    template_img_raw = cv2.imread(template_path, cv2.IMREAD_COLOR)
    template_img = standardize_image_size(template_img_raw)

    # 2. Find bounding boxes for BOTH images
    try:
        # Get the master grid size and coordinates from the Template
        tx, ty, tw, th = get_form_bounding_box(template_img, "Blank Template")
        
        # Get the grid location from the messy Target form
        rx, ry, rw, rh = get_form_bounding_box(deskewed_cv_img, "Filled Target")
        
    except Exception as e:
        print(f"Bounding box extraction failed: {e}. Returning raw standardized image.")
        return standardize_image_size(deskewed_cv_img)

    # 3. Crop the target form to its grid (with tiny padding so we don't cut off ink)
    pad = 5
    rx_pad = max(0, rx - pad)
    ry_pad = max(0, ry - pad)
    rw_pad = min(deskewed_cv_img.shape[1] - rx_pad, rw + (pad*2))
    rh_pad = min(deskewed_cv_img.shape[0] - ry_pad, rh + (pad*2))
    
    target_crop = deskewed_cv_img[ry_pad:ry_pad+rh_pad, rx_pad:rx_pad+rw_pad]

    # 4. Squeeze the target crop to match the TEMPLATE'S grid width and height
    # This prevents the "zoomed in" error!
    target_crop_resized = cv2.resize(target_crop, (tw, th), interpolation=cv2.INTER_CUBIC)

    # 5. Create a blank white canvas of our Universal Size (2550x3300)
    aligned_img = np.ones((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) * 255

    # 6. PASTE the resized target onto the blank canvas EXACTLY where the template grid is
    aligned_img[ty:ty+th, tx:tx+tw] = target_crop_resized
    
    # Show the final ghost overlay
    if DEBUG_MODE:
        show_alignment_overlay(aligned_img, template_img)
        
    return aligned_img
