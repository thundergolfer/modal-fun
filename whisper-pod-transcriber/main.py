"""
whisper-pod-transcriber uses OpenAI's Whisper modal to do speech-to-text transcription
of podcasts.
"""
import os

from typing import Any, NamedTuple

from loguru import logger

import modal

import podcast


app_image = modal.DebianSlim().pip_install(
    [
        "https://github.com/openai/whisper/archive/5d8d3e75a4826fe5f01205d81c3017a805fc2bf9.tar.gz",
        "jiwer",
        "gql[all]~=3.0.0a5",
        "pandas",
        "loguru==0.6.0",
        "torchaudio==0.12.1",
    ]
)
stub = modal.Stub("whisper-pod-transcriber", image=app_image)


def load_dataset():
    import torch
    import torchaudio
    import whisper

    class LibriSpeech(torch.utils.data.Dataset):
        """
        A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
        It will drop the last few seconds of a very small portion of the utterances.
        """

        def __init__(self, device, split="test-clean"):
            self.dataset = torchaudio.datasets.LIBRISPEECH(
                root=os.path.expanduser("~/.cache"),
                url=split,
                download=True,
            )
            self.device = device

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, item):
            audio, sample_rate, text, _, _, _ = self.dataset[item]
            assert sample_rate == 16000
            audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
            mel = whisper.log_mel_spectrogram(audio)

            return (mel, text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LibriSpeech(split="test-clean", device=device)
    return torch.utils.data.DataLoader(dataset, batch_size=16)


def evaluate(data):
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer()
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    print(f"WER: {wer * 100:.2f} %")


@stub.function(
    image=app_image,
    secret=modal.ref("podchaser"),
    gpu=True,
)
def run():
    import pandas as pd
    import numpy as np
    import whisper
    from tqdm import tqdm

    model = whisper.load_model("base.en")
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    # predict without timestamps for short-form transcription
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    hypotheses = []
    references = []
    print("Loading dataset")
    loader = load_dataset()

    print("Running inference")
    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    print(data)
    print("Evaluating model performance.")
    evaluate(data)




@stub.function(
    image=app_image,
    secret=modal.ref("podchaser"),
)
def transcribe():
    from gql import gql
    podcast_id = "1582975"  # NYT Ezra Klein Show
    client = podcast.create_podchaser_client()
    episodes = podcast.fetch_episodes_data(gql, client, podcast_id)
    print(episodes)


if __name__ == "__main__":
    with stub.run() as app:
        print(f"Modal app ID -> {app.app_id}")
        # run()
        transcribe()
