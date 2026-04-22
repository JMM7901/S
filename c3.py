def get_grid_mask(image, debug_name="Image"):
    """Extracts only horizontal and vertical lines to create a feature mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Define line length
    line_min_length = max(image.shape[1], image.shape[0]) // 50
    
    # Extract lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_min_length))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_length, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    
    # Merge and dilate to create thick, easily readable intersections
    grid_mask = cv2.addWeighted(v_lines, 0.5, h_lines, 0.5, 0.0)
    _, grid_mask = cv2.threshold(grid_mask, 50, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    grid_mask = cv2.dilate(grid_mask, kernel, iterations=1)
    
    if DEBUG_MODE: show_debug_image(f"DEBUG: Grid Mask for {debug_name}", grid_mask)
    return grid_mask


def preprocess_and_align(target_path, template_path, max_features=20000, match_percent=0.10):
    # 1. Load and Standardize Size
    pil_img = Image.open(target_path)
    deskewed_cv_img = deskew_image(pil_img)
    target_img = standardize_image_size(deskewed_cv_img)
    
    template_img_raw = cv2.imread(template_path, cv2.IMREAD_COLOR)
    template_img = standardize_image_size(template_img_raw)

    # 2. Generate the Skeleton Masks
    try:
        target_mask = get_grid_mask(target_img, "Filled Target Form")
        template_mask = get_grid_mask(template_img, "Blank Template")
    except Exception as e:
        print(f"Mask Generation Failed: {e}")
        return target_img

    # 3. Detect Features (STRICTLY ON THE GRID INTERSECTIONS)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    
    # The 'mask' parameter blinds the AI to everything except the grid lines
    keypoints1, descriptors1 = orb.detectAndCompute(target_gray, mask=target_mask)
    keypoints2, descriptors2 = orb.detectAndCompute(template_gray, mask=template_mask)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Alignment Failed: Could not find grid intersections.")

    # 4. Match the Grid Intersections
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = sorted(matcher.match(descriptors1, descriptors2, None), key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * match_percent)]

    if len(good_matches) < 20: 
        raise ValueError(f"Alignment Failed: Only {len(good_matches)} grid points matched.")

    # --- DEBUG: See the Grid Mapping ---
    if DEBUG_MODE:
        match_img = cv2.drawMatches(target_img, keypoints1, template_img, keypoints2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        show_debug_image("DEBUG: Grid-to-Grid Connections", match_img)
    # -----------------------------------

    # 5. Calculate Full-Page Homography 
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # We use a strict RANSAC threshold (3.0) to ignore any weird false matches
    h_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 3.0)
    
    if h_matrix is None:
        raise ValueError("Alignment Failed: Homography matrix is None.")

    # 6. Apply Warp
    aligned_img = cv2.warpPerspective(target_img, h_matrix, (TARGET_WIDTH, TARGET_HEIGHT))
    
    show_alignment_overlay(aligned_img, template_img)
    return aligned_img
