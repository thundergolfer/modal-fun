<h1 align="center">thundergolfer.com email subscribers app</h1>

Want to build an email list of people who want to be notified of new blog posts you publish?
This app can help :)

I like freemium email newsletter services such as [Buttondown.email](https://buttondown.email/), but I didn't
won't set up another Zapier integration just to hook up with my existing blog RSS feed.

This serverless application deploys as a simple FastAPI web app and SQLite DB, all on Modal. It uses GMail's
API to drive emails, which should work nicely up to around 1000 subscribers if you don't spam.

## Use this yourself

If you want to deploy this as the automated email subscription functionality
for your own personal blog, you'll just need two things:

1. A [Modal](https://modal.com) account
2. A Gmail account

If you have those two things, follow this [how-to blog post](https://TODO.com) (TODO) which will
help with setting up GMail authentication and deploying the Python code on Modal.

## Development

### Running app

The _first_ time run the app you'll need to setup the DB with the `setup_db` function.

```python
if __name__ == "__main__":
    with stub.run():
        setup_db.call()
    print("Done!")
```

From then on, the DB file is stored persistently on a Modal shared volume, and you can just do:

```sh
cd thundergolferdotcom-email-subs/
python3 -m email_subs.main
```

This will run the web app, and live-reload changes you make to the `.py` files.

### Testing

**Unit**

Basic testing of the datastore logic.

```sh
cd thundergolferdotcom-email-subs/
python3 -m pytest
```

**Integration**

_Coming soon_. I'd like to have a test that goes end-to-end, from checking an RSS
feed to sending an email to a test email account.

## Deploy

```sh
cd thundergolferdotcom-email-subs/
modal app deploy email_subs.main
```
