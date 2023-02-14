"""Load html from files, clean up, split, ingest into Weaviate."""
import os
from pathlib import Path

# I created a Weaviate cluster in the following way:
#
# 1. Created an account at weaviate.io; verified my email.
# 2. Clicked "Create a cluster" in the weaviate.io UI.
# 3. Selected:
#   subscription tier: sandbox
#   weaviate version: v.1.17.3
#   enable OIDC authentication: false (this data is not private)


def ingest_data(weaviate_url: str, openai_api_key: str, docs: list[str]):
    import weaviate
    from langchain.text_splitter import CharacterTextSplitter

    metadatas = [{"source": "https://thundergolfer.com/about"} for _ in docs]

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    documents = text_splitter.create_documents(docs, metadatas=metadatas)

    # WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    # os.environ["OPENAI_API_KEY"]
    client = weaviate.Client(
        url=weaviate_url,
        additional_headers={"X-OpenAI-Api-Key": openai_api_key},
    )

    client.schema.delete_all()  # drop ALL data

    client.schema.get()
    schema = {
        "classes": [
            {
                "class": "Paragraph",
                "description": "A written paragraph",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "002",
                        "type": "text",
                    }
                },
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "The content of the paragraph",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": "content",
                    },
                    {
                        "dataType": ["text"],
                        "description": "The link",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": True,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": "source",
                    },
                ],
            },
        ]
    }

    client.schema.create(schema)

    with client.batch as batch:
        for text in documents:
            batch.add_data_object(
                {"content": text.page_content, "source": str(text.metadata["source"])},
                "Paragraph",
            )
