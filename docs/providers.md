# Model providers

Planned support, not an implemented integration.

CIRCUIT uses configurable OpenAI-compatible endpoints plus an extension boundary for other protocols. OpenAPI is an API-description specification, not the model-provider protocol intended here.

Each named provider has a protocol, base URL, model ID, optional credential environment variable, timeout, and explicit capabilities. Generation and embedding providers are independent. Provider names are user-defined, not a hardcoded whitelist. Keep remote endpoint selection explicit and never silently send memory to a fallback provider.

Reserve separate adapters for Chat Completions, Responses, and embeddings. Compatibility with one endpoint does not establish compatibility with another. Streaming, tools, structured output, and usage reporting need capability checks and protocol contract fixtures. A model-list response is not proof of feature support.

OpenAI recommends Responses for new OpenAI integrations and continues to support Chat Completions. See the [official migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses). Third-party compatibility must be verified separately; no vendors are certified by this scaffold.

Shared transport/configuration belongs in circuit-providers. circuit-embed retains the embedding domain interface; circuit-agent retains the generation domain interface. Neither exposes wire-specific response types to the application.

Changing the answer model does not require rebuilding the vector index. Changing the embedding model, dimensions, or normalization requires a compatible new embedding/index artifact. Do not mix incompatible vector spaces.

Planned contract tests cover custom endpoints, missing credentials, non-streaming and streaming output, tool calls, cancellation, rate limits, unsupported features, and invalid embedding shapes. Nothing has been tested against a live provider yet.
