# QQRTC Bridge

This document describes the project-side QQ realtime-call bridge. It is intentionally limited to public-safe integration details and does not include local captures, QQ accounts, private paths, or reverse-engineering logs.

## Purpose

The QQRTC bridge lets an external LLBot/PMHQ-side plugin connect to this project, report QQNT realtime-call events, and receive call-control commands requested by model tools.

The bridge is designed as a small local WebSocket server:

- This project runs the WebSocket server.
- The external LLBot/PMHQ plugin connects as a client.
- The plugin sends call events into the project.
- The project sends control commands back to the plugin.

Media transport is out of scope for this bridge. It does not implement audio capture, TTS playback, ASR, virtual sound cards, or media stream routing.

## Configuration

Enable the bridge in `config.yaml`:

```yaml
qqrtc:
  enabled: true
  host: "127.0.0.1"
  port: 8776
  secret_token: ""
  event_buffer_size: 200
  command_timeout: 10
```

`secret_token` is optional for local development, but should be set if the server is reachable by anything outside the local trusted environment.

## Project Components

The project-side implementation is split into:

- `src/qqrtc/server.py`: WebSocket server, plugin registration, event buffering, command dispatch.
- `src/tools/get_qqrtc_calls.py`: model tool for reading plugin status, active calls, and recent events.
- `src/tools/control_qqrtc_call.py`: model tool for sending call-control commands such as `call`, `accept`, `reject`, `hangup`, and `join`.
- `src/tools/debug_qqrtc_pmhq_call.py`: latent/debug tool for sending raw PMHQ calls through the connected plugin while reverse engineering.
- `src/lifecycle.py`, `src/main.py`, and `src/app_state.py`: server lifecycle and runtime state wiring.

## Plugin Protocol

The external plugin first registers:

```json
{
  "type": "register",
  "plugin_id": "llbot-pmhq-qqrtc",
  "secret_token": "",
  "capabilities": {
    "events": true,
    "commands": true,
    "raw_call": true
  }
}
```

The server replies with:

```json
{
  "type": "register_ack",
  "accepted": true
}
```

Events are sent by the plugin as:

```json
{
  "type": "event",
  "event": {
    "post_type": "notice",
    "notice_type": "qqrtc_call",
    "sub_type": "incoming_sharp_video_push",
    "session_id": "...",
    "caller_id": "...",
    "callee_id": "...",
    "peer_id": "...",
    "direction": "incoming"
  }
}
```

Commands are sent by the project as:

```json
{
  "type": "command",
  "command_id": "...",
  "action": "call",
  "parameters": {
    "peer_id": "...",
    "session_id": "",
    "timeout_ms": 10000
  }
}
```

The plugin should respond with:

```json
{
  "type": "command_result",
  "command_id": "...",
  "ok": true,
  "status": "completed"
}
```

## Current Status

Implemented:

- Project-side QQRTC WebSocket server.
- Plugin registration and heartbeat.
- Event buffering and status querying.
- Model-facing tools for querying events and dispatching call-control commands.
- Raw PMHQ debug command path for reverse-engineering work.

Not yet complete:

- The final QQNT AVSDK command payloads for actually initiating, accepting, rejecting, and hanging up calls.
- The external plugin must still map `call` / `accept` / `reject` / `hangup` to real LLBot/PMHQ calls before the control tool can perform those actions.

## Reverse-Engineering Boundary

Normal QQ bot actions in this project typically call public or semi-public adapter APIs exposed by NapCat, LLoneBot, or OneBot-compatible protocol endpoints.

QQNT realtime voice calls do not currently have a standard public OneBot action. The bridge therefore only provides the project-side transport and model-tool surface. The external LLBot/PMHQ plugin is responsible for implementing the QQNT-specific control mapping.

During local investigation, the likely PMHQ entry point was identified as an AVSDK service command that accepts a command type and protobuf payload. That detail is intentionally not hardcoded into this project-side bridge; it belongs in the external plugin's control map.

## Safety Notes

Do not commit:

- Local LLBot plugin packages.
- PMHQ trace logs.
- QQNT reverse-engineering captures.
- QR codes.
- QQ account identifiers used during testing.
- Private local paths or tokens.

The repository `.gitignore` includes patterns for common QQRTC/PMHQ local artifacts, but sensitive files should still be reviewed before pushing.
