"""
Description: This file contains the FastAPI application that serves as the main entry point for the pipeline.
It includes endpoints to generate AI art or upload custom art and place it on a wall image.
The application is optimized to run efficiently on a GPU.
The pipeline uses the maskrcnn_resnet50_fpn_v2 and ZhengPeng7/BiRefNet models for segmentation and the Azure OpenAI DALL.E 3 model for image generation.
"""

# Importing required libraries
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForImageSegmentation
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from fastapi.responses import StreamingResponse
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from PIL import Image
from typing import List
from pydantic import BaseModel
import numpy as np
import os
import torch
import utils
import segmentation
import generation
import asyncio
import io
import zipfile

# Loading the environment variables
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
APP_API_KEY = os.getenv("APP_API_KEY")
if not AZURE_ENDPOINT:
    raise ValueError("AZURE_ENDPOINT is not set!")
if not AZURE_OPENAI_API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY is not set!")
if not APP_API_KEY:
    raise ValueError("APP_API_KEY is not set!")

# Define the device
device = torch.device("cuda")

# OpenAI image client
image_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01")

# OpenAI language client
text_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-12-01-preview")

remover_model = AutoModelForImageSegmentation.from_pretrained("./pretrained", trust_remote_code=True)
remover_model.load_state_dict(torch.load("remover_v2_GPU.pth", map_location=device))
torch.set_float32_matmul_precision(["highest"][0])
remover_model.to(device).eval().half()

rcnn_model = maskrcnn_resnet50_fpn_v2()
rcnn_model.load_state_dict(torch.load("./maskrcnn_v2.pth", map_location=device))
rcnn_model.to(device)
rcnn_model.eval()

# Generate & Segment Request schema
class GenerateSegmentRequest(BaseModel):
    api_key: str
    image_url: str
    prompt: str
    box: List[float]
    tags: List[str] = [""]
    n: int = 4

# Segment Request schema
class SegmentRequest(BaseModel):
    api_key: str
    wall_image: str
    box: List[float]
    image_urls: List[str]

# Generate Request schema
class GenerateRequest(BaseModel):
    api_key: str
    prompt: str
    size: str = "1024x1024"
    tags: List[str] = [""]
    n: int = 4

# Creating the FastAPI app instance
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defining the middleware
@app.middleware("http")
async def add_logging_and_error_handling(request: Request, call_next):
    """
    Middleware for logging all incoming requests and catching unhandled exceptions.
    Logs HTTP method and URL, and gracefully handles errors by returning appropriate HTTP status codes.
    """
    try:
        response = await call_next(request)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise e

# Validating requests
def validate_request(api_key, box):
    """
    Validates the API key and the bounding box format.
    Ensures that the API key matches the expected one and that the box is a list of 4 float values between 0 and 1.
    Raises HTTP exceptions for invalid inputs.
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required.")
    if api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    if not box or len(box) != 4:
        raise HTTPException(status_code=400, detail="Box must be 4 float values between 0 and 1.")
    for val in box:
        if not isinstance(val, float) or val < 0 or val > 1:
            raise HTTPException(status_code=400, detail="Invalid box coordinates.")
    if box[0] > box[2] or box[1] > box[3]:
        raise HTTPException(status_code=400, detail="Invalid box order.")

# Helper: segmentation
async def segment_image(rcnn_segmentor, remover_segmentor, wall, box):
    """
    Performs object segmentation on the input wall image.
    Combines the background remover and Mask R-CNN predictions to get final object masks.
    Returns None if no objects are found.
    """
    remover_mask = await remover_segmentor.predict_masks(wall, 0.5)
    final_masks = await rcnn_segmentor.predict_masks(wall, remover_mask, 0.1, box=box)
    if final_masks is None or final_masks.size == 0:
        # log_warning("No objects found during segmentation.")
        return np.zeros_like(wall)
    return final_masks

# Helper: generation
async def generate_images(generator, n):
    tasks = [utils.generate_and_fetch(generator) for _ in range(n)]
    results = await asyncio.gather(*tasks)
    return results

# Helper: process art on a wall
def process_wall_and_arts(wall, arts, box, masks):
    """
    Processes the wall image by placing each art image in the specified box region.
    Applies lighting and texture blending, then composites with segmented objects.
    Returns a list of final processed images.
    """
    box_width, box_height, x_min, y_min = utils.get_box_coordinates(wall, box)
    final_images = []
    for art in arts:
        background_np = np.array(wall)
        art_np = np.array(art)
        h, w_img = background_np.shape[:2]
        box_percent = [x_min / w_img, y_min / h, (x_min + box_width) / w_img, (y_min + box_height) / h]
        wall_art = utils.apply_lighting_and_texture(background_np, art_np, box_percent)
        final = utils.return_cropped_objects(wall_art, masks) if masks is not None else Image.fromarray(wall_art)
        final_images.append(final)
    return final_images

# Helper: image response
def create_zip_from_images(images: List[Image.Image]) -> io.BytesIO:
    """
    Creates a zip file containing a collection of images in PNG format.
    The function accepts a list of images, converts each image to PNG format, 
    and writes them into a zip file. The zip file is created in-memory and returned
    as a binary stream.
    Parameters:
    images: List[Image.Image]
        A list of PIL Image objects to include in the zip file.
    Returns:
    io.BytesIO
        A binary stream representing the zip file containing all provided images.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, img in enumerate(images):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            zip_file.writestr(f"image_{i}.png", img_bytes.read())
    zip_buffer.seek(0)
    return zip_buffer

# API endpoint to generate and place AI-generated art on a wall image
@app.post("/generate-on-wall/")
async def generate_on_wall(req: GenerateSegmentRequest):
    """
    Endpoint to generate AI art and place it on a user-provided wall image.
    - Validates request.
    - Downloads and processes wall image.
    - Asynchronously generates art and segments the wall image.
    - Places each generated art on the wall with visual enhancements.
    - Returns masked objects on the wall.
    Returns a list of image byte arrays as a zip file.
    """
    # Validate input parameters
    validate_request(req.api_key, req.box)

    # Fetch the input wall image
    wall = await utils.fetch_image(req.image_url)
    if wall is None:
        raise HTTPException(status_code=400, detail="Invalid image URL or image could not be fetched.")

    # Extract box dimensions for placing the generated art
    box_width, box_height, *_ = utils.get_box_coordinates(wall, req.box)
    size = utils.get_best_size(box_width, box_height)

    # prompt engineering
    prompt = utils.prompt_engineering(req.prompt, req.tags)
    text_generator = generation.GeneratePrompt(text_client)
    final_prompt = await text_generator.generate_prompt(prompt)

    # Initialize segmentation and image generation classes
    segmentors = (
        segmentation.MaskRCNN(rcnn_model, device),
        segmentation.BgRemover(remover_model, device)
    )
    generator = generation.GenerateImage(image_client, "dall-e-3", final_prompt, size, "standard", "natural", 1)

    # Run segmentation and image generation asynchronously
    loop = asyncio.get_event_loop()
    masks_task = loop.run_in_executor(None, lambda: asyncio.run(segment_image(*segmentors, wall=wall, box=req.box)))
    arts_task = generate_images(generator, req.n)

    masks, arts = await asyncio.gather(masks_task, arts_task)

    final_images = process_wall_and_arts(wall, arts, req.box, masks)

    zip_buffer = create_zip_from_images(final_images)

    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=images.zip"
    })

@app.post("/segment-wall/")
async def segment_wall(req: SegmentRequest):
    """
    Endpoint to place user-provided art images onto a wall image.
    - Validates request.
    - Downloads the wall image.
    - Downloads all provided art images.
    - Segments the wall image.
    - Places each art image on the wall with visual enhancements.
    Returns a ZIP file containing the final composited images.
    """
    validate_request(req.api_key, req.box)

    # Fetch wall image
    wall = await utils.fetch_image(req.wall_image)
    if wall is None:
        raise HTTPException(status_code=400, detail="Invalid wall image URL")

    # Fetch all art images from URLs
    arts = []
    for url in req.image_urls:
        art = await utils.fetch_image(url)
        if art is None:
            raise HTTPException(status_code=400, detail=f"Invalid art image URL: {url}")
        arts.append(art.convert("RGB"))

    # Initialize segmentors
    segmentors = (
        segmentation.MaskRCNN(rcnn_model, device),
        segmentation.BgRemover(remover_model, device)
    )

    # Process images and return result
    final_images = process_wall_and_arts(wall, arts, req.box, segmentors)

    zip_buffer = create_zip_from_images(final_images)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=images.zip"
    })

@app.post("/generate-art/")
async def generate_art(req: GenerateRequest):
    """
    Endpoint to generate AI art only (no wall placement).
    - Validates an API key and box format.
    - Builds the prompt from user input.
    - Generates n art images using DALL-E.
    - Returns them in a ZIP file.
    """
    # Validate an API key only (box is not needed but used in schema)
    if not req.api_key:
        raise HTTPException(status_code=401, detail="API key is required.")
    if req.api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    # prompt engineering
    prompt = utils.prompt_engineering(req.prompt, req.tags)
    text_generator = generation.GeneratePrompt(text_client)
    final_prompt = await text_generator.generate_prompt(prompt)
    size = req.size

    # Initialize generator and generate images
    generator = generation.GenerateImage(image_client, "dall-e-3", final_prompt, size, "standard", "natural", 1)
    arts = await generate_images(generator, req.n)
    arts = list(arts)

    # Return images in ZIP file
    zip_buffer = create_zip_from_images(arts)

    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=generated_art.zip"
    })