import cv2
import numpy as np
import PIL.Image
import torch

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25


def gd_caption(text_prompt):
    text = text_prompt.lower().strip()
    text = text.replace(" and ", ", ")
    parts = [p.strip().rstrip(".") for p in text.split(",") if p.strip()]
    return ". ".join(parts) + "."


def detect(pil_image, text_prompt, gd_processor, gd_model, device):
    caption = gd_caption(text_prompt)
    inputs = gd_processor(images=pil_image, text=caption, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = gd_model(**inputs)
    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[pil_image.size[::-1]],
    )[0]
    bboxes = [list(map(float, b)) for b in results["boxes"].cpu().tolist()]
    labels = [str(lbl) for lbl in results["labels"]]
    return bboxes, labels


def draw_detection_overlay(pil_image, bboxes, labels):
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    for (x1, y1, x2, y2), label in zip(bboxes, labels):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1]), (0, 255, 0), -1)
        cv2.putText(image, label, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def rescale_image(image, px=512, padding=0):
    height, width, _ = image.shape
    if [height, width].index(max([height, width])) == 0:
        factor = px / height
        height = px
        width = int(width * factor)
    else:
        factor = px / width
        width = px
        height = int(height * factor)
    image_resized = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    padded_image = np.zeros((padded_height, padded_width, image.shape[2]), dtype=np.uint8)
    x_offset = (padded_width - width) // 2
    y_offset = (padded_height - height) // 2
    padded_image[y_offset : y_offset + height, x_offset : x_offset + width] = image_resized
    return padded_image


def add_outline(image, stroke_size, outline_color):
    if image.shape[-1] != 4:
        raise ValueError("Input image must have an alpha channel (4 channels).")
    outlined_image = image.copy()
    binary = (image[:, :, 3] > 127).astype(np.uint8)

    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smooth_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, smooth_kernel)

    kernel_size = max(int(stroke_size * 0.2) * 2 + 1, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel)
    raw_outline = dilated - binary

    bg = (1 - binary).astype(np.uint8)
    wide_bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel)
    outline = raw_outline * wide_bg

    final_alpha = np.maximum(binary, outline)
    outlined_image[:, :, 3] = final_alpha * 255

    for c in range(4):
        outlined_image[:, :, c] = np.where(
            outline == 1, outline_color[c], outlined_image[:, :, c]
        )
    return outlined_image


def extract_bounding_box(image, bbox):
    if bbox:
        min_x, min_y, max_x, max_y = bbox
        return image[min_y : max_y + 1, min_x : max_x + 1]
    return None


def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    for b in contours[1:]:
        x_t, y_t, w_t, h_t = cv2.boundingRect(b)
        x1 = min(x1, x_t)
        y1 = min(y1, y_t)
        x2 = max(x2, x_t + w_t)
        y2 = max(y2, y_t + h_t)
    return [x1, y1, x2, y2]


def segment(input_path, text_prompt, sam_predictor, gd_processor, gd_model, device, detection_overlay_out=None):
    pil_image = PIL.Image.open(input_path).convert("RGB")
    np_image = np.array(pil_image)

    bboxes, labels = detect(pil_image, text_prompt, gd_processor, gd_model, device)
    if detection_overlay_out is not None:
        cv2.imwrite(detection_overlay_out, draw_detection_overlay(pil_image, bboxes, labels))
    if not bboxes:
        return False

    sam_predictor.set_image(np_image)
    union_mask = None
    for x1, y1, x2, y2 in bboxes:
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        with torch.inference_mode():
            masks, _, _ = sam_predictor.predict(
                box=box[None, :],
                multimask_output=False,
            )
        mask = masks[0].astype(np.uint8)
        union_mask = mask if union_mask is None else np.maximum(union_mask, mask)

    if union_mask is None:
        return False

    alpha = (union_mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

    image = cv2.imread(input_path)
    image = cv2.merge((image, alpha))
    return image


def make_sticker(input_path, output_path, text_prompt, input_then_path, sam_predictor, gd_processor, gd_model, device, detection_overlay_out=None):
    if "then" in text_prompt:
        text_prompt_then = text_prompt.split("then")[0]
        text_prompt = text_prompt.split("then")[-1]
        image = segment(input_path, text_prompt_then, sam_predictor, gd_processor, gd_model, device)
        if isinstance(image, bool) and image is False:
            return False
        mask = image[:, :, 3] != 0
        bbox = get_bbox_from_mask(mask)
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        original = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(input_then_path, original[y1:y2, x1:x2])
        input_path = input_then_path

    image = segment(input_path, text_prompt, sam_predictor, gd_processor, gd_model, device, detection_overlay_out=detection_overlay_out)
    if isinstance(image, bool) and image is False:
        return False

    mask = image[:, :, 3] != 0
    bbox = get_bbox_from_mask(mask)
    if bbox is None:
        return False
    image = extract_bounding_box(image, bbox)
    image = rescale_image(image, padding=30)
    image = add_outline(image, 50, (255, 255, 255, 255))
    image = rescale_image(image, padding=0)
    cv2.imwrite(output_path, image)
    return True
