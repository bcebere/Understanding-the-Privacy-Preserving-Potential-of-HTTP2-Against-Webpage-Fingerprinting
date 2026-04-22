# stdlib
import asyncio
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

# third party
from playwright.async_api import async_playwright

workspace = Path("data")
workspace.mkdir(parents=True, exist_ok=True)

bin_dir = workspace / "bin"
bin_dir.mkdir(parents=True, exist_ok=True)

client_trace_dir = workspace / "client_trace"
client_trace_dir.mkdir(parents=True, exist_ok=True)

server_trace_dir = workspace / "server_trace"
server_trace_dir.mkdir(parents=True, exist_ok=True)


async def save_responses(url, reuse_context=False, headless=True):
    hashed_path = hashlib.sha1(url.encode())
    resource_path = hashed_path.hexdigest()

    if (server_trace_dir / f"{resource_path}.json").exists():
        print(" >>> already cached")
        return

    async with async_playwright() as p:

        # Launch the browser
        if reuse_context:
            context = await p.chromium.launch_persistent_context(
                user_data_dir="~/.config/chromium/",  # Path to your existing user data directory
                ignore_https_errors=True,
                headless=headless,
            )
        else:
            browser = await p.chromium.launch(
                headless=headless
            )  # Set headless=False if you want to see the browser

            # Create a new context with caching disabled
            context = await browser.new_context(
                ignore_https_errors=True
            )  # Add any required options here

        # Set the browser context to bypass the cache
        page = await context.new_page()

        # List to store response data
        server_db = {}
        client_db = []

        # Event listener for each request to capture the start time
        request_start_times = {}
        page.on(
            "request",
            lambda request: request_start_times.__setitem__(request.url, time.time()),
        )

        # Event listener to capture responses
        page.on(
            "response",
            lambda response: asyncio.create_task(
                process_response(response, client_db, server_db, request_start_times)
            ),
        )

        # Go to the specified URL
        # await page.goto(url, wait_until="networkidle")
        await page.goto(url, wait_until="domcontentloaded")

        # Save the responses to a JSON file
        print(f" >>> saving {resource_path}")
        with open(client_trace_dir / f"{resource_path}.json", "w") as file:
            json.dump(client_db, file, indent=2)

        with open(server_trace_dir / f"{resource_path}.json", "w") as file:
            json.dump(server_db, file, indent=2)

        await context.close()


async def process_response(response, client_db, server_db, request_start_times):
    try:
        # request data
        request = response.request
        request_headers = request.headers  # Get request headers

        # response data
        response_headers = response.headers
        status = response.status
        url = response.url

        # Get Content-type
        content_type = response_headers.get("content-type", "")

        # Calculate network duration
        start_time = request_start_times.get(url, None)
        duration = 0
        if start_time:
            duration = time.time() - start_time

        # Save response data
        try:
            body = await response.body()  # Get the raw binary data
        except BaseException:
            body = b""

        url_comps = urlparse(url)
        url_local = url.split(f"{url_comps.scheme}://{url_comps.netloc}")[1]

        hashed_path = hashlib.sha1(url.encode())
        resource_path = hashed_path.hexdigest()

        filename = bin_dir / resource_path

        # Save the binary data to a file
        with open(filename, "wb") as binary_file:
            binary_file.write(body)

        client_db.append(
            {
                "url": url,
                "url_local": url_local,
                "headers": request_headers,
            }
        )

        server_db[url_local] = {
            "body_path": str(filename),
            "url": url,
            "url_local": url_local,
            "status": status,
            "content_type": content_type,
            "headers": response_headers,
            "timeout_s": duration,
        }

    except Exception as e:
        print(f"Error processing response: {e}")


def capture(url_to_capture, reuse_context=False):
    # Run the save_responses function
    asyncio.run(save_responses(url_to_capture, reuse_context=reuse_context))
