## Computing Cultural Literacy - Community stats

This small app gathers user-submitted feedback on my _Computing Cultural Literacy_ list from the
blog post of the same name.

This app just includes a serverless web API backed by a persistent SQLite DB file. It accepts data
over an API endpoint and writes it to an append-only SQLite table. Another API endpoint is used to
summarize the ingested data.

### Development

```bash
modal serve
```

```bash
curl -v 'https://thundergolfer--comp-lit-stats-web-dev.modal.run/add' -X POST -H 'Content-Type: application/json' -d '{"left": ["Spinning disk"], "right": [], "trash": []}'
```

### Deployment

```bash
modal deploy
```
