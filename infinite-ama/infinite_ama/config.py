"""
Application configuration. If you want to run this application
for yourself, you should only have to update the values in this file.
"""
from typing import Optional, NamedTuple


class UserSettings(NamedTuple):
    user_name: str
    chatbot_homepage: str
    reddit_username: Optional[str] = None
    hackernews_username: Optional[str] = None
    rss_feed_url: Optional[str] = None


# Override this config to your own personal details.
USER_SETTINGS = UserSettings(
    user_name="Jonathon",
    chatbot_homepage="https://thundergolfer.com",
    reddit_username="thundergolfer",
    hackernews_username="thundergolfer",
    rss_feed_url="https://thundergolfer.com/feed.xml",
)
