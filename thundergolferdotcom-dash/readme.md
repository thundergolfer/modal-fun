## thundergolferdotcom-dash

> Life stats with Modal webhooks

See the blog post to learn how to build this into your own website: [thundergolfer.com/modal/webhooks/dashboard/2022/11/13/personal-dashboard-with-modal-webhooks/](https://thundergolfer.com/modal/webhooks/dashboard/2022/11/13/personal-dashboard-with-modal-webhooks/).

### Development

Run the application in live-reloading mode to iterate:

```
python3 main.py
 ️️⚡️ Serving... hit Ctrl-C to stop!
└── Watching /Users/jonathonbelotti/Code/thundergolfer/modal-fun/thundergolferdotcom-dash/main.py.
✓ Initialized.
✓ Created objects.
├── 🔨 Created create_spotify_refresh_token.
├── 🔨 Mounted /Users/jonathonbelotti/Code/thundergolfer/modal-fun/thundergolferdotcom-dash/main.py at /root
├── 🔨 Created request_spotify_top_tracks.
├── 🔨 Created request_goodreads_reads.
├── 🔨 Created about_me.
└── 🔨 Created web => https://thundergolfer-thundergolferdotcom-about-page-web.modal.run
⠋ Running app...
```

### Deployment

```bash
modal deploy main.py
```
