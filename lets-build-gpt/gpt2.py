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
    .run_commands(  # remove the NVIDIA banner from showing on container startup
        "rm /opt/nvidia/entrypoint.d/10-banner.sh",
        "rm /opt/nvidia/entrypoint.d/12-banner.sh",
        "rm /opt/nvidia/entrypoint.d/15-container-copyright.txt",
        "rm /opt/nvidia/entrypoint.d/30-container-license.txt",
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


# -----------------------------------------------------------------------------
# Main conversion function

def convert(filepath, output, push_to_hub=False, out_dtype="bfloat16"):
    import numpy as np
    import torch
    from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

    # -----------------------------------------------------------------------------
    # Tensor functions for both bfloat16 (from int16) and normal float32
    # Both return float32 tensors

    def tensor_bf16(data_int16, transpose=False):
        if transpose:
            data_int16 = data_int16.transpose(1,0)
        return torch.tensor(data_int16).view(torch.bfloat16).to(torch.float32)

    def tensor_fp32(data_float32, transpose=False):
        if transpose:
            data_float32 = data_float32.transpose(1,0)
        return torch.tensor(data_float32).view(torch.float32)
    print(f"Converting model {filepath} to {output} in {out_dtype} format and pushing to Hugging Face: {push_to_hub}")

    f = open(filepath, 'rb')
    # Read in our header, checking the magic number and version
    # version 3 = fp32, padded vocab
    # version 5 = bf16, padded vocab
    model_header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if model_header[0] != 20240326:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    version = model_header[1]
    if not version in [3, 5]:
        print("Bad version in model file")
        exit(1)

    # Load in our model parameters
    maxT = model_header[2].item() # max sequence length
    V = model_header[3].item() # vocab size
    L =  model_header[4].item() # num layers
    H = model_header[5].item() # num heads
    C = model_header[6].item() # channels
    Vp = model_header[7].item() # padded vocab size

    print(f"{version=}, {maxT=}, {V=}, {Vp=}, {L=}, {H=}, {C=}")

    # Define the shapes of our parameters
    shapes = {
        'wte': (Vp, C),
        'wpe': (maxT, C),
        'ln1w': (L, C),
        'ln1b': (L, C),
        'qkvw': (L, 3 * C, C),
        'qkvb': (L, 3 * C),
        'attprojw': (L, C, C),
        'attprojb': (L, C),
        'ln2w': (L, C),
        'ln2b': (L, C),
        'fcw': (L, 4 * C, C),
        'fcb': (L, 4 * C),
        'fcprojw': (L, C, 4 * C),
        'fcprojb': (L, C),
        'lnfw': (C,),
        'lnfb': (C,),
    }

    # Load in our weights given our parameter shapes
    dtype = np.float32 if version == 3 else np.int16
    w = {}
    for key, shape in shapes.items():
        num_elements = np.prod(shape)
        data = np.frombuffer(f.read(num_elements * np.dtype(dtype).itemsize), dtype=dtype)
        w[key] = data.reshape(shape)
        # The binary file saves the padded vocab - drop the padding back to GPT2 size
        if shape[0] == Vp:
            w[key] = w[key].reshape(shape)[:(V-Vp), :]
    # Ensure the file is fully read and then close
    assert f.read() == b''
    f.close()

    # Map to our model dict, the tensors at this stage are always fp32
    mk_tensor = {
        3 : tensor_fp32,
        5 : tensor_bf16,
    }[version]
    model_dict = {}
    model_dict['transformer.wte.weight'] = mk_tensor(w['wte'])
    model_dict['transformer.wpe.weight'] = mk_tensor(w['wpe'])
    model_dict['lm_head.weight'] = model_dict['transformer.wte.weight'] # Tie weights
    for i in range(L):
        model_dict[f'transformer.h.{i}.ln_1.weight'] = mk_tensor(w['ln1w'][i])
        model_dict[f'transformer.h.{i}.ln_1.bias'] = mk_tensor(w['ln1b'][i])
        model_dict[f'transformer.h.{i}.attn.c_attn.weight'] = mk_tensor(w['qkvw'][i], True)
        model_dict[f'transformer.h.{i}.attn.c_attn.bias'] = mk_tensor(w['qkvb'][i])
        model_dict[f'transformer.h.{i}.attn.c_proj.weight'] = mk_tensor(w['attprojw'][i], True)
        model_dict[f'transformer.h.{i}.attn.c_proj.bias'] = mk_tensor(w['attprojb'][i])
        model_dict[f'transformer.h.{i}.ln_2.weight'] = mk_tensor(w['ln2w'][i])
        model_dict[f'transformer.h.{i}.ln_2.bias'] = mk_tensor(w['ln2b'][i])
        model_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = mk_tensor(w['fcw'][i], True)
        model_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = mk_tensor(w['fcb'][i])
        model_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = mk_tensor(w['fcprojw'][i], True)
        model_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = mk_tensor(w['fcprojb'][i])
    model_dict['transformer.ln_f.weight'] = mk_tensor(w['lnfw'])
    model_dict['transformer.ln_f.bias'] = mk_tensor(w['lnfb'])

    # Create a GPT-2 model instance, in the requested dtype
    config = GPT2Config(vocab_size = V,
                        n_positions = maxT,
                        n_ctx = maxT,
                        n_embd = C,
                        n_layer = L,
                        n_head = H)
    model = GPT2LMHeadModel(config)
    if out_dtype == "bfloat16":
        model = model.to(torch.bfloat16)

    # Set the model dict and save
    model.load_state_dict(model_dict)
    model.save_pretrained(output, max_shard_size="5GB", safe_serialization=True)

    # Copy over a standard gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output)

    if push_to_hub:
        print(f"Uploading {output} to Hugging Face")
        model.push_to_hub(output)
        tokenizer.push_to_hub(output)

def spin(output):
    print("Taking the exported model for a spin...")
    print('-'*80)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForCausalLM.from_pretrained(output, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    tokens = tokenizer.encode("During photosynthesis in green plants", return_tensors="pt")
    tokens = tokens.to('cuda')
    output = model.generate(tokens, max_new_tokens=64, repetition_penalty=1.3)
    samples = tokenizer.batch_decode(output)
    for sample in samples:
        print('-'*30)
        print(sample)

# -----------------------------------------------------------------------------
@app.function(
    image=(
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
        .apt_install("git")
        .pip_install(
            # required to build flash-attn
            "ninja",
            "packaging",
            "wheel",
            "torch",
            # further deps
            "accelerate",
            "transformers",
            "huggingface_hub",
        )
        .run_commands(  # add flash-attn
            "pip install flash-attn --no-build-isolation"
        )
    ),
    volumes={
        OUT_MOUNT_PATH: out_volume,
    },
    timeout=10 * 60,
    gpu="a100",
)
def export_hf():
    """
    Script to convert GPT2 models from llm.c binary format to Hugging Face

    It can optionally upload to your account on Hugging Face if you have the CLI:
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login
    """
    # input_file: The name of the llm.c model.bin file
    # output: The Hugging Face output model directory
    # push_to_hub: Push the model to your Hugging Face account
    # out_dtype: Output as either float32 or bfloat16 (default)
    input_file = pathlib.Path(OUT_MOUNT_PATH, "log_gpt2_1558M/model_00000004.bin")
    output_dir = pathlib.Path(OUT_MOUNT_PATH, "hf_format_models")
    convert(input_file, output_dir, push_to_hub=False, out_dtype="bfloat16")
    spin(output_dir)


@app.local_entrypoint()
def main():
    run_train_node.remote(steps=4)
