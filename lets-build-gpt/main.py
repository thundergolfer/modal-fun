import os
import pathlib
import subprocess
import time

import modal

app = modal.App("lets-build-gpt")

DEPS = ["jupyter", "torch"]
image = (
    modal.Image.debian_slim()
    .pip_install("uv==0.2.26") # use uv because it's faster than pip!
    .run_commands(f"uv pip install --system --compile {' '.join(DEPS)}")
)
app = modal.App(
    image=image
)
volume = modal.Volume.from_name(
    "lets-build-gpt", create_if_missing=True
)

CACHE_DIR = pathlib.Path("/root/cache")
JUPYTER_TOKEN = "friendlyjordies"  # Change me to something non-guessable!
SESSION_TIMEOUT = 60 * 120  # 2 hours
GPT_DEV_NB = pathlib.Path(__file__).parent / "gpt-dev.ipynb"


@app.function(volumes={CACHE_DIR: volume})
def seed_volume():
    import urllib.request

    output_file = pathlib.Path(CACHE_DIR, "tinyshakespeare.txt")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    if not output_file.exists():
        urllib.request.urlretrieve(url, output_file)
        print(f"Downloaded Shakespeare text to {output_file}")
    else:
        print(f"Shakespeare text already downloaded to {output_file}")
    volume.commit()


# This is all that's needed to create a long-lived Jupyter server process in Modal
# that you can access in your Browser through a secure network tunnel.
# This can be useful when you want to interactively engage with Volume contents
# without having to download it to your host computer.


@app.function(
    concurrency_limit=1,
    volumes={CACHE_DIR: volume},
    timeout=SESSION_TIMEOUT,
)
def run_jupyter(timeout: int, local_data: dict[str, bytes]):

    for path, data in local_data.items():
        pathlib.Path(path).write_bytes(data)

    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


if __name__ == "__main__":
    with modal.enable_output(), app.run():
        # Write some images to a volume, for demonstration purposes.
        seed_volume.remote()
        # Pass our local notebook state to the remote container.
        # It will sync this with the notebook file in the Volume.
        sync_local_to_remote = {
            CACHE_DIR / "gpt-dev.ipynb": GPT_DEV_NB.read_bytes()
        }
        # Run the Jupyter Notebook server
        run_jupyter.remote(timeout=SESSION_TIMEOUT, local_data=sync_local_to_remote)

    # Grab back any changes we made to the notebook in the remote container.
    # These changes will have been saved into the modal.Volume.
    remote_to_local = {
        "gpt-dev.ipynb": GPT_DEV_NB
    }
    for vol_path, local_path in remote_to_local.items():
        data = b""
        for chunk in volume.read_file(vol_path):
            data += chunk
        local_path.write_bytes(data)
        print(f"Updated {local_path}")
    
