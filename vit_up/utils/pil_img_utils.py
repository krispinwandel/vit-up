import math
from typing import List, Optional, Tuple
from PIL import Image, ImageOps, ImageDraw, ImageFont
import warnings
import numpy as np
import matplotlib.cm as cm
import torch
import colorsys


def get_color(i, n_colors=None):
    """
    Returns a distinct RGB tuple (0-255) using Golden Ratio spacing.
    n_colors is optional here because the Golden Ratio works infinitely.
    """
    # The Golden Ratio conjugate
    phi_conjugate = 0.618033988749895

    # Use the golden ratio to "jump" across the color wheel
    # This prevents neighboring indices from having neighboring colors
    hue = (i * phi_conjugate) % 1.0

    # We can also slightly jitter saturation and value for more variety
    saturation = 0.6 + (i % 3) * 0.1  # Cycles between 0.6, 0.7, 0.8
    value = 0.8 + (i % 2) * 0.1  # Cycles between 0.8, 0.9

    rgb_fractional = colorsys.hsv_to_rgb(hue, saturation, value)

    return tuple(int(c * 255) for c in rgb_fractional)


def img_to_tensor_imgnet(img_pil):
    img_tensor = (
        torch.from_numpy(np.array(img_pil))
        .type(torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )
    img_net_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    img_net_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - img_net_mean) / img_net_std
    return img_tensor


def pad_image(img, px, py, ensure_square=False):
    width, height = img.size
    new_width = width + 2 * px
    new_height = height + 2 * py
    if ensure_square:
        max_side = max(new_width, new_height)
        new_width = max_side
        new_height = max_side
    # Create new image with desired size and background color
    new_img = Image.new(img.mode, (new_width, new_height), color=0)
    # Paste original image onto the center
    new_img.paste(img, (px, py))
    return new_img


def get_square_paddings(width, height):
    if width == height:
        return 0, 0
    elif width > height:
        py = (width - height) // 2
        return 0, py
    else:
        px = (height - width) // 2
        return px, 0


def get_target_aspect_ratio_paddings(width, height, target_aspect_ratio):
    current_aspect_ratio = width / height
    if current_aspect_ratio == target_aspect_ratio:
        return 0, 0
    elif current_aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio
        new_height = int(width / target_aspect_ratio)
        py = (new_height - height) // 2
        return 0, py
    else:
        # Image is taller than target aspect ratio
        new_width = int(height * target_aspect_ratio)
        px = (new_width - width) // 2
        return px, 0


def pad_image_to_square(img):
    px, py = get_square_paddings(*img.size)
    return pad_image(img, px, py)


def pad_image_to_size(img, target_size):
    width, height = img.size
    target_width, target_height = target_size
    px = max(0, (target_width - width) // 2)
    py = max(0, (target_height - height) // 2)
    return pad_image(img, px, py)


def pad_image_to_aspect_ratio(img, target_aspect_ratio):
    """
    Args:
        img: PIL.Image
        target_aspect_ratio: width / height
    Returns:
        PIL.Image padded to target aspect ratio
    """
    width, height = img.size
    px, py = get_target_aspect_ratio_paddings(width, height, target_aspect_ratio)
    return pad_image(img, px, py)


def crop_img(img, bbox_xyxy):
    img_cropped = img.crop(bbox_xyxy)
    return img_cropped


def heatmap_to_rgb(heatmap, colormap="viridis", heat_min=None, heat_max=None):
    """
    Converts a heatmap (2D NumPy array) to an RGB image.

    Args:
        heatmap (np.ndarray): 2D array representing the heatmap.
        colormap (str): Name of the matplotlib colormap to use (default: 'viridis').

    Returns:
        Image.Image: PIL RGB image.
    """
    if not isinstance(heatmap, np.ndarray):
        raise ValueError("Heatmap must be a NumPy array.")
    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be a 2D array.")

    # Normalize the heatmap to the range [0, 1]
    if heat_min is None:
        heat_min = heatmap.min()
    if heat_max is None:
        heat_max = heatmap.max()
    heatmap_normalized = (heatmap - heat_min) / (heat_max - heat_min)

    # Apply the colormap
    colormap_func = cm.get_cmap(colormap)
    heatmap_colored = colormap_func(heatmap_normalized)  # Returns RGBA values

    # Convert to RGB (drop the alpha channel)
    heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    # Convert to PIL Image
    return Image.fromarray(heatmap_rgb)


def interpolate_images(img1, img2, alpha=0.5):
    """
    Interpolates between two images using a blending factor.

    Args:
        img1 (Image.Image or np.ndarray): The first image (PIL Image or NumPy array).
        img2 (Image.Image or np.ndarray): The second image (PIL Image or NumPy array).
        alpha (float): Blending factor, where 0.0 corresponds to `img1` and 1.0 to `img2`.

    Returns:
        Image.Image: The interpolated image as a PIL Image.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be in the range [0, 1].")

    # Convert images to NumPy arrays if they are PIL Images
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)

    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Both images must have the same dimensions.")

    # Perform interpolation
    interpolated = (1 - alpha) * img1 + alpha * img2
    interpolated = np.clip(interpolated, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(interpolated)


def apply_mask_black(img, mask):
    """
    Sets pixels to black where mask is False/0.

    Args:
        img (Image.Image or np.ndarray): Input image.
        mask (np.ndarray): Boolean or 0/1 mask, shape (H, W).

    Returns:
        Image.Image: Masked image as PIL Image.
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    mask = np.asarray(mask).astype(bool)
    img_masked = img.copy()
    img_masked[~mask] = 0

    return Image.fromarray(img_masked)


def crop_img_with_rotation_pil(
    img_pil, bbox, angle_deg, resample=Image.BILINEAR, fill=0
):
    """
        Args:
            img_pil:  PIL.Image
            bbox:     (xmin, ymin, xmax, ymax)
            angle_from PIL import Image, ImageDraw, ImageFont
    import mathdeg: rotation angle
            resample:  e.g. Image.BILINEAR, Image.NEAREST, Image.BICUBIC
            fill:      pad fill value (default 0)
    """
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    s = max(w, h)

    assert s % 2 == 0, "bbox width and height must be even"

    # Larger square to avoid black corners after rotation
    diam = int(math.ceil(math.sqrt(2) * s))
    diam = diam + (diam % 2)
    pad = (diam - s) // 2

    # Zero-pad original image
    img_pil_pad = ImageOps.expand(img_pil, border=(pad, pad, pad, pad), fill=fill)

    # Adjust bbox after padding
    xmin_pad = xmin
    ymin_pad = ymin
    xmax_pad = xmax + 2 * pad
    ymax_pad = ymax + 2 * pad

    # Crop expanded region
    img_crop = img_pil_pad.crop((xmin_pad, ymin_pad, xmax_pad, ymax_pad))

    # Rotate in-place (no expand)

    if angle_deg > 0:
        img_crop = img_crop.rotate(angle_deg, resample=resample)

    # Center-crop back to original w × h
    left = (img_crop.width - w) // 2
    top = (img_crop.height - h) // 2
    return img_crop.crop((left, top, left + w, top + h))


def draw_circle(img, center, radius, color=(255, 0, 0), width=3):
    """
    Draw a circle on a PIL image.

    Args:
        img: PIL.Image
        center: (x, y)
        radius: int
        color: RGB tuple
        width: stroke width

    Returns:
        PIL.Image with circle drawn.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    x, y = center
    bbox = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bbox, outline=color, width=width)
    return img


def concat_images(
    img_list,
    target_width=None,
    mode="row",
    pad=0,
    pad_color=(255, 255, 255),
    interpolate_resample=Image.LANCZOS,
):
    """
    Resize images to target_width (preserving aspect ratio)
    and concatenate horizontally ("row") or vertically ("col")
    with optional padding between images.

    Args:
        img_list: list[PIL.Image]
        target_width: int
        mode: "row" or "col"
        pad: int, pixels between images
        pad_color: RGB tuple for padding / background
        interpolate_resample: resampling method for resizing (e.g. Image.LANCZOS)

    Returns:
        PIL.Image
    """
    assert mode in ("row", "col")
    assert pad >= 0
    assert len(pad_color) == 3

    if target_width is None:
        target_width = max(img.width for img in img_list)

    if not img_list:
        raise ValueError("img_list must not be empty")

    # Resize images to common width
    resized = []
    for img in img_list:
        w, h = img.size
        new_h = int(h * (target_width / w))
        resized.append(img.resize((target_width, new_h), interpolate_resample))

    n = len(resized)

    if mode == "row":
        total_width = target_width * n + pad * (n - 1)
        max_height = max(im.height for im in resized)

        canvas = Image.new("RGB", (total_width, max_height), pad_color)
        x_off = 0
        for i, im in enumerate(resized):
            canvas.paste(im, (x_off, 0))
            x_off += im.width
            if i < n - 1:
                x_off += pad
        return canvas

    else:  # column
        total_height = sum(im.height for im in resized) + pad * (n - 1)

        canvas = Image.new("RGB", (target_width, total_height), pad_color)
        y_off = 0
        for i, im in enumerate(resized):
            canvas.paste(im, (0, y_off))
            y_off += im.height
            if i < n - 1:
                y_off += pad
        return canvas


def add_description_to_image(
    img: Image.Image,
    description: str,
    margin: int = 4,
    font_size: int = 20,
    placement: str = "bottom",  # "bottom" or "top"
    text_align: str = "left",  # "left", "center", "right"
    bg_color: str = "black",
    font_color: str = "white",
    text_box_height_ratio: float | None = None,
    textbox_width: int | None = None,
    textbox_height: int | None = None,
    overlay: bool = True,
    font_path: str | None = None,
    interpolate_resample: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    """
    Render description text into a fixed-size pixel textbox.

    - If textbox_width is set, resize image width to match it (keep aspect ratio).
    - Wrap text when line width reaches (textbox_width - 2 * margin).
    - Vertically center wrapped text in textbox_height.
    - Clip overflowing text naturally to textbox bounds.

    text_box_height_ratio is deprecated and ignored.
    """

    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Image must have non-zero size.")
    if margin < 0:
        raise ValueError("margin must be >= 0.")
    if font_size <= 0:
        raise ValueError("font_size must be > 0.")

    if text_box_height_ratio is not None:
        warnings.warn(
            "text_box_height_ratio is deprecated and ignored. "
            "Use textbox_height instead.",
            UserWarning,
            stacklevel=2,
        )

    if textbox_width is not None and textbox_width <= 0:
        raise ValueError("textbox_width must be > 0.")
    if textbox_height is not None and textbox_height <= 0:
        raise ValueError("textbox_height must be > 0.")

    if textbox_width is None:
        W = orig_w
        base_img = img
    else:
        W = int(textbox_width)
        if W != orig_w:
            resized_h = max(1, int(round(orig_h * (W / orig_w))))
            base_img = img.resize((W, resized_h), resample=interpolate_resample)
        else:
            base_img = img

    H = base_img.height
    max_text_width = W - 2 * margin
    if max_text_width < 1:
        raise ValueError("textbox_width - 2 * margin must be >= 1.")

    # Split on explicit newlines, then wrap each line by pixel width.
    raw_lines = description.split("\n") if description else [""]

    # Load font
    def load_font(sz: int) -> ImageFont.FreeTypeFont:
        if font_path is not None:
            return ImageFont.truetype(font_path, sz)
        for fp in ("arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(fp, sz)
            except OSError:
                continue
        return ImageFont.load_default()

    font = load_font(font_size)

    # Measure each line's width and height using real bboxes
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    def text_width_px(text: str) -> int:
        sample_text = text if text else " "
        bbox = dummy_draw.textbbox((0, 0), sample_text, font=font)
        return int(bbox[2] - bbox[0])

    def split_long_token(token: str, max_w: int) -> list[str]:
        parts: list[str] = []
        current = ""
        for ch in token:
            candidate = f"{current}{ch}"
            if current and text_width_px(candidate) > max_w:
                parts.append(current)
                current = ch
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts if parts else [""]

    def wrap_line_by_width(text: str, max_w: int) -> list[str]:
        if text == "":
            return [""]

        words = text.split(" ")
        wrapped: list[str] = []
        current = ""

        for word in words:
            if current:
                candidate = f"{current} {word}"
            else:
                candidate = word

            if candidate and text_width_px(candidate) <= max_w:
                current = candidate
                continue

            if current:
                wrapped.append(current)
                current = ""

            if word == "":
                continue

            if text_width_px(word) <= max_w:
                current = word
            else:
                long_parts = split_long_token(word, max_w)
                wrapped.extend(long_parts[:-1])
                current = long_parts[-1]

        if current:
            wrapped.append(current)

        return wrapped if wrapped else [""]

    lines: list[str] = []
    for raw_line in raw_lines:
        lines.extend(wrap_line_by_width(raw_line, max_text_width))

    line_widths = []
    line_heights = []

    for line in lines:
        # For empty lines, use a sample with ascender/descender so height is safe
        sample_text = line if line.strip() else "Agpq"
        bbox = dummy_draw.textbbox((0, 0), sample_text, font=font)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]
        line_widths.append(line_w)
        line_heights.append(line_h)

    total_text_h = sum(line_heights)
    line_spacing = margin
    if len(lines) > 1:
        total_text_h += line_spacing * (len(lines) - 1)

    if textbox_height is None:
        box_h = max(1, int(math.ceil(total_text_h + 2 * margin)))
    else:
        box_h = int(textbox_height)

    final_text_box = Image.new("RGB", (W, box_h), bg_color)
    draw = ImageDraw.Draw(final_text_box)

    y = (box_h - total_text_h) / 2.0
    for line, line_h in zip(lines, line_heights):
        txt = line
        line_w = text_width_px(txt)

        if text_align == "left":
            x = margin
        elif text_align == "center":
            x = (W - line_w) // 2
        elif text_align == "right":
            x = W - line_w - margin
        else:
            raise ValueError("text_align must be 'left', 'center', or 'right'.")

        if txt:
            draw.text((x, y), txt, fill=font_color, font=font)
        y += line_h + line_spacing

    # Compose with original image
    if overlay:
        new_img = base_img.copy()
        if placement == "bottom":
            y0 = H - box_h
        elif placement == "top":
            y0 = 0
        else:
            raise ValueError("placement must be 'bottom' or 'top'.")
        new_img.paste(final_text_box, (0, y0))
        return new_img
    else:
        new_h = H + box_h
        new_img = Image.new("RGB", (W, new_h), bg_color)
        if placement == "bottom":
            new_img.paste(base_img, (0, 0))
            new_img.paste(final_text_box, (0, H))
        elif placement == "top":
            new_img.paste(final_text_box, (0, 0))
            new_img.paste(base_img, (0, box_h))
        else:
            raise ValueError("placement must be 'bottom' or 'top'.")
        return new_img


def create_img_grid(img_list, max_cols=4, target_width=300):
    rows = []
    for i in range(0, len(img_list), max_cols):
        row_imgs = img_list[i : i + max_cols]
        row_img = concat_images(row_imgs, target_width=target_width, mode="row")
        rows.append(row_img)
    grid_img = concat_images(rows, target_width=target_width * max_cols, mode="col")
    return grid_img


def draw_points(
    h,
    w,
    points,
    point_colors: Optional[List[Tuple[int, int, int]]] = None,
    radius=2.0,
    scale_factor=8,
):
    """
    Renders points onto an anti-aliased canvas using supersampling.

    Args:
        h, w: Final image dimensions.
        points: List of (x, y) coordinates.
        radius: Point radius in final pixel units.
        scale_factor: The multiplier for supersampling (e.g., 8).

    Returns:
        PIL.Image: The rendered grid.
    """
    # Create the high-res internal canvas
    big_w, big_h = w * scale_factor, h * scale_factor
    big_radius = radius * scale_factor

    # Draw at high resolution
    big_img = Image.new("RGB", (big_w, big_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(big_img)

    if point_colors is None:
        point_colors = [(255, 255, 255)] * len(points)

    for i, pt in enumerate(points):
        # Scale the points up to the big canvas
        bx, by = pt[0] * scale_factor, pt[1] * scale_factor

        # Define bounding box for the ellipse
        bbox = [bx - big_radius, by - big_radius, bx + big_radius, by + big_radius]
        draw.ellipse(bbox, fill=point_colors[i])

    # Downscale using LANCZOS (high-quality interpolation)
    return big_img.resize((w, h), resample=Image.Resampling.LANCZOS)
