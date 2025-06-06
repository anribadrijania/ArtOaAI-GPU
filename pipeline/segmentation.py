from torchvision import transforms
import torch
import numpy as np
import cv2


def transformer_for_rcnn(image, device):
    image_np = np.array(image)
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensors = transform(image).unsqueeze(0).to(device)

    return image_np, input_tensors


def preprocess_image(image, device):
    """Preprocess the image while maintaining an aspect ratio."""
    original_size = image.size  # Save the original size (width, height)

    image_size = (1024, 1024)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device).half()
    return input_tensor, original_size  # Return both image and original size


class MaskRCNN:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    async def predict_masks(self, image, bg_mask, threshold=0.5, overlap_thresh=0.5,
                            containment_thresh=0.8, box=None, fill_thresh=0.5):
        ignore_labels = [48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 60, 75, 76, 77, 87, 89, 90]
        image_np, input_tensors = transformer_for_rcnn(image, self.device)
        height, width = image_np.shape[:2]

        # Convert relative box to absolute coords
        x_min, y_min, x_max, y_max = box
        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)
        box_area = (x_max - x_min) * (y_max - y_min)

        with torch.no_grad():
            outputs = self.model(input_tensors)

        scores = outputs[0]['scores'].cpu().numpy()
        masks = outputs[0]['masks'].squeeze().cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        passed_masks = []

        for i in range(len(scores)):
            if labels[i] in ignore_labels:
                continue
            if scores[i] < threshold:
                continue

            mask = (masks[i] > 0.5).astype(np.uint8)

            # Step 1: Per-mask area check inside the box
            mask_in_box = mask[y_min:y_max, x_min:x_max]
            fill_ratio = mask_in_box.sum() / box_area
            if fill_ratio >= fill_thresh:
                continue  # Skip this mask

            # Step 2: Overlap and containment checks
            intersection = np.logical_and(bg_mask, mask).sum()
            union = np.logical_or(bg_mask, mask).sum()
            iou = intersection / union if union != 0 else 0

            rcnn_area = mask.sum()
            if rcnn_area == 0:
                continue
            contained_ratio = intersection / rcnn_area

            if iou < overlap_thresh or contained_ratio < containment_thresh:
                passed_masks.append(mask)

        # Step 3: Combined mask check inside the box
        if box and passed_masks:
            combined_mask = np.zeros_like(passed_masks[0])
            for mask in passed_masks:
                # Only combine those that touch the box area
                if np.any(mask[y_min:y_max, x_min:x_max]):
                    combined_mask = np.logical_or(combined_mask, mask)

            combined_fill_ratio = combined_mask[y_min:y_max, x_min:x_max].sum() / box_area
            if combined_fill_ratio >= fill_thresh:
                # Filter out all passed masks that intersect with the box
                final_masks = [m for m in passed_masks if not np.any(m[y_min:y_max, x_min:x_max])]
            else:
                final_masks = passed_masks
        else:
            final_masks = passed_masks

        # Combine final masks with bg_mask
        for mask in final_masks:
            bg_mask = cv2.bitwise_or(bg_mask, mask * 255)

        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]

        final_image_np = np.dstack((image_np, bg_mask))
        return final_image_np


class BgRemover:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    async def predict_masks(self, image, threshold=0.5):
        input_tensor, original_size = preprocess_image(image, self.device)
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = (preds[0].squeeze() > threshold).float()

        # Convert mask to PIL Image & Resize to original dimensions
        mask_pil = transforms.ToPILImage()(pred)
        mask = mask_pil.resize(image.size)

        # Return binary mask as NumPy array
        mask_np = np.array(mask)
        bg_mask = (mask_np > 128).astype(np.uint8)
        bg_mask = bg_mask.astype(np.uint8) * 255
        return bg_mask

