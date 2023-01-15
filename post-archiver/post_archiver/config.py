from typing import Optional, NamedTuple

class UserSettings(NamedTuple):
    reddit_username: Optional[str] = None
    hackernews_username: Optional[str] = None
    rss_feed_url: Optional[str] = None


# Override this config to your own personal details.
USER_SETTINGS = UserSettings(
    reddit_username="thundergolfer",
    hackernews_username="thundergolfer",
    rss_feed_url="https://thundergolfer.com/feed.xml",
)
