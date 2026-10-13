# Using a different API

localsearch talks to any OpenAI-compatible chat completions endpoint, not just
Jan. Point it elsewhere with `--base-url`:

```bash
./localsearch-run --base-url http://localhost:11434/v1   # Ollama
./localsearch-run --base-url http://localhost:1234/v1    # LM Studio
./localsearch-run --base-url http://localhost:8080/v1    # llama.cpp server
```

The URL is remembered in `~/.localsearch/config.json`, so you only pass it
once. You can also edit that file directly, including `api_key` for servers
that want one:

```json
{
  "base_url": "http://localhost:11434/v1",
  "api_key": "not-needed",
  "context_size": 8192
}
```

`--model` should match a model id that server lists at `/v1/models`. Embedding
models (`nomic-embed`, `sentence-transformer`, ...) are hidden from the picker
since they can't answer questions.

Set `context_size` to what the model is actually loaded with. It decides how
many notes fit into the final TLDR prompt.
