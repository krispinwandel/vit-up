from PIL import Image, ImageDraw
import torch


def to_pil_rgb(rgb_tensor: torch.Tensor) -> Image.Image:
    rgb = rgb_tensor.detach().cpu()
    if rgb.dtype.is_floating_point:
        if float(rgb.max()) <= 1.0:
            rgb = rgb * 255.0
        rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    else:
        rgb = rgb.to(torch.uint8)
    return Image.fromarray(rgb.numpy())


def overlay_points(
    base_img: Image.Image,
    points_xz_normalized: torch.Tensor,
    point_colors: torch.Tensor | None = None,
) -> Image.Image:
    if point_colors is None:
        point_colors = torch.tensor(
            [[255, 64, 64]], device=points_xz_normalized.device
        )  # default red color
        point_colors = point_colors.expand(
            points_xz_normalized.shape[0], -1
        )  # (n_points, 3)
    out = base_img.convert("RGB").copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)
    r = max(4, int(round(0.012 * min(w, h))))
    for i, point_xz in enumerate(points_xz_normalized):
        cx = int(point_xz[0].item() * w)
        cz = int(point_xz[1].item() * h)
        draw.ellipse(
            (cx - r, cz - r, cx + r, cz + r),
            fill=tuple(point_colors[i].cpu().numpy().astype(int)),
            outline=(255, 255, 255),
            width=2,
        )
    return out


def overlay_heatmap(base_img: Image.Image, heatmap: torch.Tensor) -> Image.Image:
    heat = pil_img_utils.heatmap_to_rgb(heatmap.detach().cpu().numpy()).resize(
        base_img.size,
        resample=Image.Resampling.NEAREST,
    )
    return Image.blend(base_img.convert("RGB"), heat.convert("RGB"), alpha=0.55)


def overlay_imgs(
    base_img: Image.Image, overlay_img: Image.Image, alpha=0.5
) -> Image.Image:
    overlay_resized = overlay_img.resize(
        base_img.size,
        resample=Image.Resampling.NEAREST,
    )
    return Image.blend(
        base_img.convert("RGB"), overlay_resized.convert("RGB"), alpha=alpha
    )
