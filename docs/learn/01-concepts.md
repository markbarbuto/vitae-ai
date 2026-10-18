# Concepts behind the design

## Origin allowlist

The browser adds the `Origin` header itself. Page JavaScript cannot change it. The backend compares it with the list the owner set in `VITAE_ALLOWED_ORIGINS`.

![Origin allowlist](images/concepts-01-origin-allowlist.png)

Another website's page is rejected, because its browser reports its own origin.

![Origin allowlist](images/concepts-02-origin-allowlist.png)

A script outside a browser can fake the origin. Per-IP rate limits are what stop it.

![Origin allowlist](images/concepts-03-origin-allowlist.png)

## How other public chatbots handle access

![How other public chatbots handle access](images/concepts-04-how-other-public-chatbots-handle-access.png)

Hosted services need a public key because one server serves many customers, and the key says which customer a request belongs to. Vitae runs one backend per owner, so there is nothing for a key to select. The origin allowlist does the same job without one.

## Chunking

![Chunking](images/concepts-05-chunking.png)

## Embeddings

Each passage becomes a list of numbers. Passages with similar meaning get similar numbers.

![Embeddings](images/concepts-06-embeddings.png)

## Search at question time

![Search at question time](images/concepts-07-search-at-question-time.png)

## Cloning versus an npm package

![Cloning versus an npm package](images/concepts-08-cloning-versus-an-npm-package.png)

During development nothing is published. The portfolio links to the local package folder.

![Cloning versus an npm package](images/concepts-09-cloning-versus-an-npm-package.png)

## Where Hugging Face fits

![Where Hugging Face fits](images/concepts-10-where-hugging-face-fits.png)
