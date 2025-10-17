import asyncio
import re
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
from KVCOMM.utils.log import logger


async def process_tasks(prompts, api_key):
    tasks = [generate_image(prompt, api_key) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error("Image generation task failed: {}", result)
            processed_results.append(None)
        else:
            processed_results.append(result)

    return processed_results


async def generate_image(prompt, api_key):
    client = AsyncOpenAI(api_key=api_key)
    image_params = {
        "model": "dall-e-3",
        "quality": "standard",
        "style": "natural",
        "n": 1,
        "size": "1024x1024",
        "prompt": prompt,
    }
    res = await client.images.generate(**image_params)
    return res.data[0].url


def extract_dimensions(url):

    matches = re.findall(r"(\d+)x(\d+)", url)

    if matches:
        width, height = matches[0]                          
        width = int(width)
        height = int(height)
        return (width, height)
    else:
        return (100, 100)


def create_alt_url_mapping(code):
    soup = BeautifulSoup(code, "html.parser")
    images = soup.find_all("img")

    mapping = {}

    for image in images:
        if not image["src"].startswith("https://placehold.co"):
            mapping[image["alt"]] = image["src"]

    return mapping


async def generate_images(code, api_key, image_cache):

    soup = BeautifulSoup(code, "html.parser")
    images = soup.find_all("img")


    alts = []
    for img in images:


        if (
            img["src"].startswith("https://placehold.co")
            and image_cache.get(img.get("alt")) is None
        ):
            alts.append(img.get("alt", None))


    alts = [alt for alt in alts if alt is not None]


    prompts = list(set(alts))


    if len(prompts) == 0:
        return code


    results = await process_tasks(prompts, api_key)


    mapped_image_urls = dict(zip(prompts, results))


    mapped_image_urls = {**mapped_image_urls, **image_cache}


    for img in images:

        if not img["src"].startswith("https://placehold.co"):
            continue

        new_url = mapped_image_urls[img.get("alt")]

        if new_url:

            width, height = extract_dimensions(img["src"])
            img["width"] = width
            img["height"] = height

            img["src"] = new_url
        else:
            logger.warning("Image generation failed for alt text: {}", img.get("alt"))



    return soup.prettify()
