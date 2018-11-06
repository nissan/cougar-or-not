from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import (
    ImageDataBunch,
    create_cnn,
    open_image,
    get_transforms,
    models,
)
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()

hb_images_path = Path("/tmp")
hb_fnames = [
    "/{}_1.jpg".format(c)
    for c in [
        'black_throated_mango_female',
        'black_throated_mango_male',
        'blue_chinned_sapphire_female',
        'blue_chinned_sapphire_male',
        'blue_tailed_emerald_female',
        'blue_tailed_emerald_male',
        'brown_violetear',
        'copper_rumped',
        'green_hermit_female',
        'green_hermit_male',
        'green_throated_mango',
        'little_hermit',
        'long_billed_starthroat',
        'ruby_topaz_female',
        'ruby_topaz_male',
        'rufous_breasted_hermit',
        'tufted_coquette_female',
        'tufted_coquette_male',
        'white_chested_emerald',
        'white_necked_jacobin_female',
        'white_necked_jacobin_male',
        'white_tailed_goldenthroat',
        'white_tailed_sabrewing'
        ]
]
hb_data = ImageDataBunch.from_name_re(
    hb_images_path,
    hb_fnames,
    r"/([^/]+)_\d+.jpg$",
    ds_tfms=get_transforms(),
    size=224,
)
hb_learner = create_cnn(hb_data, models.resnet34)
hb_learner.model.load_state_dict(
    torch.load("stage-2-34_3.pth", map_location="cpu")
)


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _,_,losses = hb_learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(hb_learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <h1>Trinidad and Tobago Hummingbird Classifier v0.01.1</h1>
        <p> This can distinguish among the following species
            <ul>
                <li>Ruby Topaz (Female)</li>
                <li>Ruby Topaz (Male)</li>
                <li>Black Throated Mango (Female)</li>
                <li>Black Throated Mango (Male)</li>
                <li>Blue Chinned Sapphire (Female)</li>
                <li>Blue Chinned Sapphire (Male)</li>
                <li>Blue Tailed Emerald (Female)</li>
                <li>Blue Tailed Emerald (Male)</li>
                <li>Brown Violetear</li>
                <li>Copper Rumped</li>
                <li>Green Throated Mango</li>
                <li>Long Billed Starthroat</li>
                <li>Tufted Coquette (Female)</li>
                <li>Tufted Coquette (Male)</li>
                <li>Green Hermit (Female)</li>
                <li>Green Hermit (Male)</li>
                <li>Little Hermit</li>
                <li>Rufous Breasted Hermit</li>
                <li>White Chested Emerald</li>
                <li>White Necked Jacobin (Female)</li>
                <li>White Necked Jacobin (Male) </li>
                <li>White Tailed Goldenthroat</li>
                <li>White Tailed Sabrewing</li>
            </ul>
        </p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
