# Browser image lazy materialization

Status: implemented in the production browser world, QQ `send_message` path,
and QQ adapter. The real-Chrome check below exercises the production registry
and materializer; only the LLM selection decision is deterministic.

## Goals

- Stop eagerly copying every browser image response into the AIcarus cache.
- Remove automatic per-image viewport clips.
- Let the model select a lightweight browser resource before original bytes are
  materialized.
- Prefer bytes already held by Chrome and avoid a second HTTP request.
- Keep browser internals out of the model prompt and QQ send contract.

## Browser observation

Chrome continues to load and render the page normally. AIcarus records a
lightweight resource descriptor for each relevant DOM image and produces one
full viewport screenshot for page understanding.

No per-image screenshot is generated when original bytes are unavailable.

### Internal resource record

The browser resource registry owns:

- `resource_ref`
- raw `source_url`
- `page_url`
- ephemeral CDP `request_id`
- ephemeral CDP `frame_id`
- observation timestamp
- alt text, viewport rectangle, and natural dimensions
- optional materialized `image_ref`

Cookies remain in the browser context. Credentials, local paths and request
headers are never projected to the model. Full model-visible URLs retain useful
query structure but redact credential-like values.

### Model-visible projection

The browser world exposes only:

- page URL and title once at page level;
- per image: `resource_ref`;
- optionally hidden, sanitized, or full `source_url`;
- alt text;
- viewport rectangle;
- natural dimensions.

It does not expose `request_id`, `frame_id`, repeated per-image `page_url`,
local paths, hashes, cookies, request headers, or cache policy details.

## Model selection and materialization

The model selects a `resource_ref`. A separate materialization operation then
tries to obtain the original encoded bytes in this order:

1. Exact observed response via `Network.getResponseBody(request_id)`.
2. Current page resource via `Page.getResourceContent(frame_id, source_url)`.
3. Chrome network stack with cache enabled.
4. Actual network transfer only when Chrome cannot satisfy the request from
   memory cache, disk cache, or validation.

The resulting response must pass protocol, redirect destination, MIME,
decodability, byte-size and pixel-count checks. Accepted raster bytes are
persisted by content hash in the immutable send-artifact store and exposed as a
sendable `image_ref`.

Failure is explicit. It never falls back to a region screenshot.

## QQ send boundary

The model may put at most four browser `resource_ref` values in one
`send_message` batch. The tool materializes them in message order and replaces
them internally with `image_ref`. The QQ adapter accepts only artifacts whose
manifest, content hash, MIME, dimensions and byte count still validate. A
viewport or element screenshot cannot cross this boundary.

## Configuration and optional confirmation

`browser_control.image_source_url` controls the model projection:

- `full` (default): useful URL structure with credential-like values redacted;
- `sanitized`: scheme, host and path only;
- `hidden`: omit the per-image URL.

`browser_control.image_send_confirmation` controls an optional second step:

- `off` (default): hard validation succeeds, then the message sends directly;
- `high_risk`: explicit semantic differences stage the whole batch for one
  normal composite round.

High-risk reasons are named rather than scored: unproven resource identity,
redirect or scheme change, MIME/content-form change, aspect-ratio change, a
small or cropped preview, and animation/multiple frames. The next round retains
all ordinary tools and temporarily adds `confirm_browser_image_send` and
`cancel_browser_image_send`. It displays at most four final immutable originals.
The target, full message payload, inbound revision, artifact refs and content
hashes are bound internally. A new inbound message, focus change, changed
artifact, explicit cancellation, or an unhandled confirmation round cancels the
batch.

Security failures never enter confirmation and always block.

## Experiment

Run:

```powershell
.\.venv\Scripts\python.exe scripts\browser_image_materialization_check.py
```

Machine-readable report:

```powershell
.\.venv\Scripts\python.exe scripts\browser_image_materialization_check.py --json
```

The fixture verifies:

- internal browser fields do not reach the simulated LLM;
- the simulated LLM chooses only by model-visible metadata and `resource_ref`;
- exact response-body extraction preserves original bytes without another HTTP
  request;
- page-resource-cache extraction preserves original bytes without another HTTP
  request;
- the production fallback order reaches Chrome's cache-enabled network stack
  without a second image transfer;
- only the selected resource is persisted;
- production and reference paths preserve byte-for-byte equality;
- failure produces no `image_ref`, artifact, or screenshot fallback.
