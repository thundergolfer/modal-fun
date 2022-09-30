import podcast

_HTML_SHELL = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Favicon -->
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔊</text></svg>">
</head>

<body class="bg-gray-50">
    {body}
</body>
</html>
"""


def html_page(title: str, body: str) -> str:
    return _HTML_SHELL.format(title=title, body=body)


def html_podcast_header(pod: podcast.PodcastMetadata) -> str:
    # Add a bit of padding between lines in HTML description.
    podcast_description_html = pod.html_description.replace("<p>", "<p class='py-1'>")
    return f"""
    <div class="mx-auto max-w-4xl mt-4 py-8 rounded overflow-hidden shadow-lg">
        <div class="px-6 py-4">
            <div class="font-bold text-xl">{pod.title}</div>
            <div class="text-gray-700 text-md">
                {podcast_description_html}
            </div>
        </div>
    </div>
    """


def html_podcast_404_page() -> str:
    body = """
<div className="mx-auto max-w-md py-16">
    <div class="flex justify-center">
        <h3>Sorry, this podcast hasn't been processed yet. Head back to the home page.</h3>
    </div>
</div>"""
    return _HTML_SHELL.format(title="Modal Podcast Transcriber | 404", body=body)


def html_episode_header(episode: podcast.EpisodeMetadata) -> str:
    # Add a bit of padding between lines in HTML description.
    episode_description_html = episode.html_description.replace(
        "<p>", "<p class='py-1'>"
    )
    return f"""
<div class="mx-auto max-w-4xl py-8 rounded overflow-hidden shadow-lg">
    <div class="px-6 py-4">
        <div class="font-bold text-l text-green-500 mb-2">{episode.show}</div>
        <div class="font-bold text-xl mb-2">{episode.title}</div>
        <div class="text-gray-700 text-sm py-4">
            {episode_description_html}
        </div>
    </div>
</div>
    """


def html_all_transcripts_header() -> str:
    return """
<div class="mx-auto max-w-4xl pt-4 pb-2 rounded overflow-hidden shadow-lg">
    <div class="px-6 py-4">
        <div class="font-bold text-center text-2xl mt-4">All Transcripts</div>
        <p>
            Every time a web user requests transcripts of a podcast, we grab a handful of the latest
            episodes and transcribe them. Click any of the transcribe episodes below to view an openai/whisper
            transcription!
        </p>
    </div>
</div>
    """


def html_episode_list(episodes) -> str:
    list_content = ""
    for ep in episodes:
        episode_li = f"""<li class="px-6 py-2 border-b border-gray-200 w-full rounded-t-lg">
            <a href="/transcripts/{ep.podcast_id}/{ep.guid_hash}" class="text-blue-700 no-underline hover:underline">
                {ep.title}
            </a> | {ep.publish_date}
        </li>
        """
        list_content += episode_li
    return f"""
<div class="mx-auto max-w-4xl py-8">
    <ul class="bg-white rounded-lg border border-gray-200 w-384 text-gray-900">
        {list_content}
    </ul>
</div>
    """


def html_transcript_list(segments) -> str:
    segments_ul_html = """<ul class="bg-white rounded-lg border border-gray-200 w-384 text-gray-900">"""
    for segment in segments:
        segment_li = f"""<li class="px-6 py-2 border-b border-gray-200 w-full rounded-t-lg">
            {segment["text"]}
        </li>
        """
        segments_ul_html += segment_li
    segments_ul_html += "</ul>"
    return f"""
<div class="mx-auto max-w-4xl py-8">
    <div class="font-bold text-xl text-blue-500 mb-2">Transcript</div>
    <div>
        {segments_ul_html}
    </div>
</div>
    """
