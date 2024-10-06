from fastapi import FastAPI, Query
from typing import Optional
from fastapi.responses import FileResponse
import pystac_client
import planetary_computer
import odc.stac
import matplotlib.pyplot as plt
import os

# Set matplotlib to use the 'Agg' backend for non-interactive environments
plt.switch_backend('Agg')

from pystac.extensions.eo import EOExtension as eo

app = FastAPI()

# Load catalog once, globally
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

def fetch_landsat_images(bbox_of_interest, time_of_interest):
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 30}},
    )

    items = search.item_collection()
    if not items:
        return None, None, None

    selected_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
    
    # Log available bands for debugging
    available_bands = list(selected_item.assets.keys())
    print(f"Available bands: {available_bands}")
    
    # Updated band list without 'lwir11'
    bands_of_interest = ["nir08", "red", "green", "blue", "qa_pixel"]
    
    data = odc.stac.stac_load(
        [selected_item], bands=bands_of_interest, bbox=bbox_of_interest
    ).isel(time=0)

    return selected_item, data, bbox_of_interest


@app.get("/generate-images")
def generate_images(
    bbox: str = Query(..., description="Bounding box in the format 'x1,y1,x2,y2'"),
    time_of_interest: Optional[str] = Query(
        "2021-01-01/2021-12-31", description="Time range in the format 'YYYY-MM-DD/YYYY-MM-DD'"
    ),
):
    try:
        # Split the 'bbox' string into four float values
        bbox_list = [float(coord) for coord in bbox.split(",")]
        
        if len(bbox_list) != 4:
            return {"error": "Bounding box must contain exactly 4 values (x1, y1, x2, y2)."}
        
    except ValueError:
        return {"error": "Bounding box values must be valid floats in the format 'x1,y1,x2,y2'."}

    # Fetch Landsat data based on bounding box and time of interest
    selected_item, data, bbox_of_interest = fetch_landsat_images(bbox_list, time_of_interest)

    if selected_item is None:
        return {"error": "No data found for the given bounding box and time range."}

    # Create image files
    natural_color_path = "natural_color_image.png"
    ndvi_image_path = "ndvi_image.png"

    # Plot Natural Color
    fig, ax = plt.subplots(figsize=(10, 10))
    data[["red", "green", "blue"]].to_array().plot.imshow(robust=True, ax=ax)
    ax.set_title(f"Natural Color, Redmond, WA")
    fig.savefig(natural_color_path)
    plt.close(fig)

    # Compute and Plot NDVI
    red = data["red"].astype("float")
    nir = data["nir08"].astype("float")
    ndvi = (nir - red) / (nir + red)

    fig, ax = plt.subplots(figsize=(14, 10))
    ndvi.plot.imshow(ax=ax, cmap="viridis")
    ax.set_title(f"NDVI, Redmond, WA")
    fig.savefig(ndvi_image_path)
    plt.close(fig)

    # Return the image files as responses
    return {
        "natural_color_image": f"/download-image/{natural_color_path}",
        "ndvi_image": f"/download-image/{ndvi_image_path}",
    }


@app.get("/download-image/{image_path}")
def download_image(image_path: str):
    return FileResponse(image_path, media_type="image/png")
