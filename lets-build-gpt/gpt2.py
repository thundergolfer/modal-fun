"""
https://github.com/karpathy/llm.c/discussions/677
"""
import os
import pathlib
import subprocess

import modal
import modal.experimental

app = modal.App("lets-reproduce-gpt2")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
os_ = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "wget", "curl")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get -y install libcudnn9-dev-cuda-12",
        "git clone https://github.com/NVIDIA/cudnn-frontend.git ~/cudnn-frontend",
        # install MPI (optional, if you intend to use multiple GPUs)
        # (you might also have to install NVIDIA NCCL if it doesn't come with your setup)
        "apt -y install openmpi-bin openmpi-doc libopenmpi-dev",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/thundergolfer/llm.c",
        "cd llm.c && git checkout 38dd4dbe",
        # download the "starter pack" (~1GB download)
        # contains GPT2-124M weights (used in tests), tokenizer, eval data .bin s
        "cd llm.c && ./dev/download_starter_pack.sh",
        # We set NO_USE_MPI=1 to disable MPI because we use the TCP as the nccl_init_method (-pi)
        # TODO(Jonathon): disabling MPI should not be necessary but `multi_gpu_config_free` in zero.cu
        # does not check that the init_method == 'mpi' and thus incorrectly calls MPI_Finalize().
        "cd llm.c && make train_gpt2cu USE_CUDNN=1 NO_USE_MPI=1",
        # Make is not handling errors by exiting with a non-zero code.
        # Ensure the above succeeded by enforcing existance of /llm.c/train_gpt2cu
        "test -f /llm.c/train_gpt2cu",
        gpu="any",
    )
)
app = modal.App(image=image)

data_volume = modal.Volume.from_name(
    "lets-reproduce-gpt2-edu_fineweb100B-vol", create_if_missing=True
)
out_volume = modal.Volume.from_name(
    "lets-reproduce-gpt2-output-vol", create_if_missing=True
)
# llm.c outputs its downloaded EDU FineWeb training data inside the repo
# under this path.
DATA_MOUNT_PATH = "/llm.c/dev/data/edu_fineweb100B"
OUT_MOUNT_PATH = "/out"
GPU_COUNT = 2
GPU = modal.gpu.H100(count=GPU_COUNT)


@app.function(volumes={DATA_MOUNT_PATH: data_volume})
def download_data(shards: int = 1):
    subprocess.run(
        ["./edu_fineweb.sh", str(shards)], check=True, cwd="/llm.c/dev/data/"
    )
    data_volume.commit()


@app.function(
    gpu=GPU,
    volumes={
        DATA_MOUNT_PATH: data_volume,
        OUT_MOUNT_PATH: out_volume,
    },
    timeout=18 * 60 * 60,
)
def train_gpt2(
    # 32k steps takes 24 hours on 8xH100
    steps: int = 32000,
):
    subprocess.run(["nvidia-smi"], check=True)

    print("and train! (wait 24 hours here)")
    args = [
        "./train_gpt2cu",
        # -i -j are training and validation splits token files, downloaded earlier with edu_fineweb.sh
        "-i",
        "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin",
        "-j",
        "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin",
        # -o is the output directory to write logs and checkpoints into
        "-o",
        str(pathlib.Path(OUT_MOUNT_PATH) / "log_gpt2_1558M"),
        # -v 250 asks to evaluate and log the validation loss every 250 steps
        "-v",
        "250",
        # -s 300000 asks to sample some tokens every 300000 steps. Because the total number of steps will be less than this, this is hacky way to turn sampling off and we will only sample a single time at the very end.
        "-s",
        "300000",
        # -g 384 sets the number of tokens to be sampled at the end to be 384
        "-g",
        "384",
        # -h 1 asks to evaluate the HellaSwag accuracy
        "-h",
        "1",
        # -b 16 sets the micro-batch size to 16 . If you are running out of memory, decrease this value, e.g. try 8, 4, 2, all the way down to 1 potentially.
        "-b",
        "16",
        # -t 1024 sets the maximum sequence length to 1024, as GPT-2 did
        "-t",
        "1024",
        # -d 1048576 asks that the total batch size be 2 to the power 20, following the GPT-3 paper hyperparameters table. The code will make sure to meet this desired total batch size and calculate the needed gradient accumulation "inner loop" steps of the optimization. For example up above, we saw that we have 8 GPUs each doing 16 X 1024 tokens, so that is 8 X 16 X 1024 = 131,072 tokens per micro-step (a single forward backward), so the code calculated gradient accumulation steps of 8 to meet the desired 1M batch size per step. i.e. it does forward+backward 8 times and then a single update.
        "-d",
        "1048576",
        # -r 0 sets recompute to zero. Recompute is a way to trade off compute and memory. If -r 1, then we recompute a piece of the forward pass (the GeLU) during backward. This means we don't have to cache it and save memory, at the cost of some more compute. So if you're running out of memory, try -r 1, or -r 2 (also recompute layernorms).
        "-r",
        "0",
        # -z 1 turns on ZeRO-1 (i.e. optimizer state sharding) across multiple GPUs. If you're training with > 1 GPU, this setting is a no-brainer and should basically always be on. On 1 GPU this setting is a no-op.
        "-z",
        "1",
        # -c 0.1 sets the weight decay to 0.1. Only (2D) weights are decayed exactly as in GPT-2, and this number comes from the GPT-3 paper
        "-c",
        "0.1",
        # -k "cosine" sets the cosine learning rate schedule, which is the default so this is a bit spurious.
        "-k",
        "cosine",
        # -l 0.0006 sets the maximum learning rate to 6e-4. The GPT-3 paper says to use 2e-4 for this model size, but here we triple and it and seems to train faster and without any issues. This wasn't tuned very carefully yet.
        "-l",
        "0.0006",
        # -q 0.1 says that we will decay the learning rate to 10% of max LR over the course of training, following GPT-3 paper.
        "-q",
        "0.1",
        # -u 700 says that we will ramp up the learning rate from 0 to max learning rate over the first 700 iterations, which at total batch size 0.5M is 350M tokens, following GPT-3 paper.
        "-u",
        "700",
        # -n 2000 asks to save model checkpoints every 2000 steps.
        "-n",
        "2000",
        # -x 32000 asks for 32K steps in total. I chose this number because it is a nice number, and just fits into 24 hours.
        "-x",
        str(steps),
        # -ge 1 sets a very recently merged gelu recompute setting for CublasLt (optional)
        "-ge",
        "1",
        # -y 1 sets the "resume" flag on. If your training for any reason crashes or hangs, you can CTRL+C and re-run this command, and it will attempt to resume the optimization. llm.c is bitwise-deterministic, so you'll get the identical result as if you didn't crash.
        "-y",
        "1",
        # -e "d48" asks to initialize, a depth 48 GPT-2 model from scratch.
        "-e",
        "d48",
    ]
    if GPU_COUNT > 1:
        args = [
            # the launch command: we're using mpi to launch GPU_COUNT processes
            # (each process runs training on 1 GPU, for e.g. 8 GPUs total on an example 8XH100 node).
            "mpirun",
            # we're running in a container, so risks of root are minimal
            "--allow-run-as-root",
            "-np",
            str(GPU_COUNT),
        ] + args

    subprocess.run(args, check=True, cwd="/llm.c")


@app.function(
    gpu=GPU,
    timeout=60 * 60,
    cloud="gcp",
    volumes={
        DATA_MOUNT_PATH: data_volume,
        OUT_MOUNT_PATH: out_volume,
    },
    _experimental_scheduler_placement=modal.scheduler_placement.SchedulerPlacement(zone="us-east4-a")
)
@modal.experimental.grouped(size=2)
def run_train_node(
    # 32k steps takes 24 hours on 8xH100
    steps: int = 32000,
):
    # subprocess.run(["nvidia-smi"], check=True)

    world_size = int(os.environ.get("MODAL_WORLD_SIZE", 1))
    container_rank = os.environ["MODAL_CONTAINER_RANK"]
    main_addr = os.environ["MODAL_MAIN_I6PN"]
    nccl_ifname = os.environ["NCCL_SOCKET_IFNAME"]
    cloud = os.environ["MODAL_CLOUD_PROVIDER"]
    print(
        f"world_size: {world_size}, container_rank: {container_rank}, main_addr: {main_addr}, nccl_ifname: {nccl_ifname}, cloud: {cloud}"
    )

    print("and train! (wait 24 hours here)")
    args = [
        "./train_gpt2cu",
        # -i -j are training and validation splits token files, downloaded earlier with edu_fineweb.sh
        "-i",
        "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin",
        "-j",
        "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin",
        # -o is the output directory to write logs and checkpoints into
        "-o",
        str(pathlib.Path(OUT_MOUNT_PATH) / "log_gpt2_1558M"),
        # -v 250 asks to evaluate and log the validation loss every 250 steps
        "-v",
        "250",
        # -s 300000 asks to sample some tokens every 300000 steps. Because the total number of steps will be less than this, this is hacky way to turn sampling off and we will only sample a single time at the very end.
        "-s",
        "300000",
        # -g 384 sets the number of tokens to be sampled at the end to be 384
        "-g",
        "384",
        # -h 1 asks to evaluate the HellaSwag accuracy
        "-h",
        "1",
        # -b 16 sets the micro-batch size to 16 . If you are running out of memory, decrease this value, e.g. try 8, 4, 2, all the way down to 1 potentially.
        "-b",
        "4",
        # -t 1024 sets the maximum sequence length to 1024, as GPT-2 did
        "-t",
        "1024",
        # -d 1048576 asks that the total batch size be 2 to the power 20, following the GPT-3 paper hyperparameters table. The code will make sure to meet this desired total batch size and calculate the needed gradient accumulation "inner loop" steps of the optimization. For example up above, we saw that we have 8 GPUs each doing 16 X 1024 tokens, so that is 8 X 16 X 1024 = 131,072 tokens per micro-step (a single forward backward), so the code calculated gradient accumulation steps of 8 to meet the desired 1M batch size per step. i.e. it does forward+backward 8 times and then a single update.
        "-d",
        "1048576",
        # -r 0 sets recompute to zero. Recompute is a way to trade off compute and memory. If -r 1, then we recompute a piece of the forward pass (the GeLU) during backward. This means we don't have to cache it and save memory, at the cost of some more compute. So if you're running out of memory, try -r 1, or -r 2 (also recompute layernorms).
        "-r",
        "0",
        # -z 1 turns on ZeRO-1 (i.e. optimizer state sharding) across multiple GPUs. If you're training with > 1 GPU, this setting is a no-brainer and should basically always be on. On 1 GPU this setting is a no-op.
        "-z",
        "1",
        # -c 0.1 sets the weight decay to 0.1. Only (2D) weights are decayed exactly as in GPT-2, and this number comes from the GPT-3 paper
        "-c",
        "0.1",
        # -k "cosine" sets the cosine learning rate schedule, which is the default so this is a bit spurious.
        "-k",
        "cosine",
        # -l 0.0006 sets the maximum learning rate to 6e-4. The GPT-3 paper says to use 2e-4 for this model size, but here we triple and it and seems to train faster and without any issues. This wasn't tuned very carefully yet.
        "-l",
        "0.0006",
        # -q 0.1 says that we will decay the learning rate to 10% of max LR over the course of training, following GPT-3 paper.
        "-q",
        "0.1",
        # -u 700 says that we will ramp up the learning rate from 0 to max learning rate over the first 700 iterations, which at total batch size 0.5M is 350M tokens, following GPT-3 paper.
        "-u",
        "700",
        # -n 2000 asks to save model checkpoints every 2000 steps.
        "-n",
        "2000",
        # -x 32000 asks for 32K steps in total. I chose this number because it is a nice number, and just fits into 24 hours.
        "-x",
        str(steps),
        # -ge 1 sets a very recently merged gelu recompute setting for CublasLt (optional)
        "-ge",
        "1",
        # -y 1 sets the "resume" flag on. If your training for any reason crashes or hangs, you can CTRL+C and re-run this command, and it will attempt to resume the optimization. llm.c is bitwise-deterministic, so you'll get the identical result as if you didn't crash.
        "-y",
        "1",
        # -e "d48" asks to initialize, a depth 48 GPT-2 model from scratch.
        "-e",
        "d48",
    ]
    if True:
        multi_node_args = [
            "-pr",
            container_rank,
            "-pi",
            "tcp",
            "-ps",
            main_addr,
            "-pg",
            str(GPU_COUNT),
            "-pn",
            str(world_size),
        ]
    else:
        multi_node_args = []

    if GPU_COUNT > 1:
        args = (
            [
                # the launch command: we're using mpi to launch GPU_COUNT processes
                # (each process runs training on 1 GPU, for e.g. 8 GPUs total on an example 8XH100 node).
                "mpirun",
                # we're running in a container, so risks of root are minimal
                "--allow-run-as-root",
                "-np",
                str(GPU_COUNT),
            ]
            + args
            + multi_node_args
        )

    subprocess.run(args, check=True, cwd="/llm.c")


@app.local_entrypoint()
def main():
    run_train_node.remote(steps=4)
