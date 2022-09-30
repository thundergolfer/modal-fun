import dataclasses
import pathlib


@dataclasses.dataclass
class ModelSpec:
    name: str
    params: str
    relative_speed: int  # Higher is faster


CACHE_DIR = "/cache"
RAW_AUDIO_DIR = pathlib.Path(CACHE_DIR, "raw_audio")
METADATA_DIR = pathlib.Path(CACHE_DIR, "metadata")
COMPLETED_DIR = pathlib.Path(CACHE_DIR, "completed")
TRANSCRIPTIONS_DIR = pathlib.Path(CACHE_DIR, "transcriptions")
SEARCH_DIR = pathlib.Path(CACHE_DIR, "search")
ASSETS_PATH = pathlib.Path(__file__).parent / "web"
# TODO: Remove
podchaser_podcast_ids = {
    "ezra_klein_nyt": 1582975,
    "ezra_klein_vox": 82327,
    "lex_fridman": 721928,
    "The Joe Rogan Experience": 10829,
}


supported_whisper_models = {
    "tiny.en": ModelSpec(name="tiny.en", params="39M", relative_speed=32),
    # Takes around 3-10 minutes to transcribe a podcast, depending on length.
    "base.en": ModelSpec(name="base.en", params="74M", relative_speed=16),
    "small.en": ModelSpec(name="small.en", params="244M", relative_speed=6),
    "medium.en": ModelSpec(name="medium.en", params="769M", relative_speed=2),
    # Very slow. Will take around 45 mins to 1.5 hours to transcribe.
    "large": ModelSpec(name="large", params="1550M", relative_speed=1),
}
