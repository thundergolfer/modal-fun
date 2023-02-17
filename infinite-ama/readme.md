<p align="center">
  <img src="https://user-images.githubusercontent.com/12058921/212502958-771f18f3-750f-4b56-8e40-37088cf2696c.png"/>
</p>

<h1 align="center">Infinite Ask-me-Anything</h1>

> Periodically archive blog posts and internet comments, and use them to train a GPT chat-bot.

![web user interface for Infinite AMA app](./hero-infinite-ama.png)

---

> 🚧 Work In Progress

The **post archiver** part of the Modal application makes it easy to continuously backup your online posts to a database.
With this backup you can do fun things:

- Full-text search, and text analytics
- Train a chat-bot
- Train a question-answering bot

The application currently supports collecting posts from the following sources:

- [Reddit](https://www.reddit.com/)
- [Hackernews](https://news.ycombinator.com/)
- RSS (which I used to collect personal blog posts from [thundergolfer.com/blog](https://thundergolfer.com/blog))

## How-to setup

1. Create a [Modal.com](https://modal.com) account, if you don't already have one.
2. Fill in the required config values at XXX
3. Create a `modal.Secret` for YYY
4. Create a `virtualenv` and install locally-required packages with `pip install -r requirements.txt`.

todo

## Development

### Running

```
modal run infinite_ama.main
```

### Testing

todo

### Deployment

```
modal deploy infinite_ama.main
```
