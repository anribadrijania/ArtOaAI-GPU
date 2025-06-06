from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import aiohttp
import cv2
import base64


async def fetch_image(url):
    """
    Fetch an image from a given URL asynchronously.

    :param url: The URL of the image.
    :return: PIL Image object if successful.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                image = Image.open(BytesIO(data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return image
            return None


def get_box_coordinates(wall, box):
    """
    Convert percentage-based bounding box coordinates into pixel values.

    :param wall: The base image.
    :param box: The bounding box in percentage values.
    :return: Pixel-based bounding box coordinates.
    """
    base_width, base_height = wall.size

    # Convert box coordinates from percentages to pixel values
    x_min = int(box[0] * base_width)
    y_min = int(box[1] * base_height)
    x_max = int(box[2] * base_width)
    y_max = int(box[3] * base_height)

    # Calculate box width and height
    box_width = x_max - x_min
    box_height = y_max - y_min

    return box_width, box_height, x_min, y_min


def get_best_size(width: int, height: int) -> str:
    """
    Determine the best image size based on aspect ratio.

    :param width: Original width of the image.
    :param height: Original height of the image.
    :return: Best matching size as a string.
    """
    sizes = [(1024, 1024), (1792, 1024), (1024, 1792)]
    box_ratio = width / height
    best_size = min(sizes, key=lambda size: abs(box_ratio - size[0] / size[1]))

    return f"{best_size[0]}x{best_size[1]}"


async def generate_and_fetch(generator):
    """
    Generate an image using the generator class and fetch the resulting image.

    :param generator: An instance of the Generate class.
    :return: PIL Image object.
    """
    prompt, custom_image = await generator.generate_image_with_revised_prompt()
    image = await fetch_image(custom_image)
    return image


def apply_lighting_and_texture(background: np.ndarray, artwork: np.ndarray, box_percent: list) -> Image.Image:
    h, w, _ = background.shape
    x_min_px = int(box_percent[0] * w)
    y_min_px = int(box_percent[1] * h)
    x_max_px = int(box_percent[2] * w)
    y_max_px = int(box_percent[3] * h)

    box_region = background[y_min_px:y_max_px, x_min_px:x_max_px]
    box_region_float = box_region.astype(np.float32) / 255.0
    illum_color = cv2.GaussianBlur(box_region_float, (51, 51), 0)  # blurred RGB lighting

    # Compute grayscale version (intensity)
    illum_gray = cv2.cvtColor(illum_color, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

    # Compute color strength: how far color is from grayscale
    color_strength = np.linalg.norm(illum_color - illum_gray, axis=2, keepdims=True)  # (H, W, 1)

    # Normalize strength between 0 (gray) and 1 (highly colored)
    max_strength = 0.2  # tweak: higher = less sensitive to color
    blend_factor = np.clip(color_strength / max_strength, 0, 1)

    # Final illumination map: blend between grayscale and colored
    illum_blend = illum_gray * (1 - blend_factor) + illum_color * blend_factor

    # Optional: clamp to avoid extreme lighting
    illum_blend = np.clip(illum_blend, 0.01, 1.9)

    box_height = y_max_px - y_min_px
    box_width = x_max_px - x_min_px

    artwork_resized = cv2.resize(artwork, (box_width, box_height))
    artwork_float = artwork_resized.astype(np.float32) / 255.0

    # Apply blended lighting
    artwork_lit = artwork_float * illum_blend
    artwork_lit = np.clip(artwork_lit, 0, 1)

    # Calculate average brightness in box region
    luminance = 0.2126 * box_region_float[..., 0] + 0.7152 * box_region_float[..., 1] + 0.0722 * box_region_float[
        ..., 2]
    avg_brightness = np.mean(luminance)

    # Define target brightness (0 = dark, 1 = bright), adjust as needed
    target_brightness = 0.8

    # Compute brightness factor (clamped for safety)
    brightness_factor = target_brightness / max(avg_brightness, 1e-4)
    brightness_factor = np.clip(brightness_factor, 0.0, 2.0)  # Prevent over/underexposure
    # Apply adaptive brightness to the full image
    artwork_lit *= brightness_factor

    # Apply subtle wall texture
    texture_scaled = cv2.resize(box_region, (box_width, box_height))
    texture_float = texture_scaled.astype(np.float32) / 255.0
    texture_overlay = texture_float * 0.00 + artwork_lit * 1.00
    texture_overlay = np.clip(texture_overlay, 0, 1)

    # Edge fade mask
    alpha_channel = create_rounded_fade_mask(box_width, box_height)

    # Blend final result into background
    result_image = background.astype(np.float32) / 255.0
    result_image[y_min_px:y_max_px, x_min_px:x_max_px, :3] = (
            result_image[y_min_px:y_max_px, x_min_px:x_max_px, :3] * (1 - alpha_channel[..., np.newaxis]) +
            texture_overlay * alpha_channel[..., np.newaxis]
    )

    result_image_uint8 = (np.clip(result_image, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(result_image_uint8)


def create_rounded_fade_mask(width, height, fade_ratio=0.02, corner_radius_ratio=0.08):
    fade_width_x = int(width * fade_ratio)  # Fade width for the horizontal dimension
    fade_width_y = int(height * fade_ratio)  # Fade width for the vertical dimension
    radius = int(min(width, height) * corner_radius_ratio)

    mask_img = Image.new("L", (width, height), 0)  # Start with a fully transparent image
    draw = ImageDraw.Draw(mask_img)

    # Draw the rounded rectangle, inset by the fade_width for both dimensions
    draw.rounded_rectangle(
        (fade_width_x, fade_width_y, width - fade_width_x, height - fade_width_y),
        radius=radius,
        fill=255  # Fill with opaque (255)
    )

    # Apply Gaussian blur to ensure a smooth fade from transparent to opaque
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(max(fade_width_x, fade_width_y)))

    # Convert the image to an array and normalize to [0, 1]
    mask_array = np.array(mask_img).astype(np.float32) / 255.0
    return mask_array


def return_cropped_objects(image, masks):
    """
    Overlay a transparent image of cropped objects onto the background image.

    :param image: The base image (background).
    :param masks: The image of cropped objects.
    :return: Modified image with objects pasted back.
    """
    try:
        cropped_objects = Image.fromarray(masks)
        image = image.convert("RGBA")
        cropped_objects = cropped_objects.convert("RGBA")

        result_image = Image.alpha_composite(image, cropped_objects)
        return result_image
    except Exception as e:
        print(f"Error: {e}")
        return None


def pil_to_binary(img):
    """
    Convert a PIL image to binary format.

    :param img: The input PIL image.
    :return: Binary representation of the image.
    """
    with BytesIO() as buffer:
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        return b64


def prompt_engineering(prompt, tags):
    """
    Generates an engineered prompt for a wall art request.

    Parameters:
    prompt (str): The original order text from the client.
    tags (list): A list of artistic styles to be included.

    Returns:
    str: The formatted prompt incorporating the client's request and styles.
    """

    styles = " Use the following artistic styles: " + ", ".join(tags) + "."

    engineered_prompt = f"{prompt}. {styles}"
    return engineered_prompt
