# OpenWebUI Functions

This directory contains repo-managed OpenWebUI Function files.

Current function:

- `kankor_status_pipe.py`
  - Adds a selectable Pipe model in OpenWebUI
  - Proxies requests to `/v1/chat/stream`
  - Turns backend `progress` SSE events into native OpenWebUI `status` updates
  - Streams backend `delta` text into the active assistant message

When running the local OpenWebUI Docker stack, this directory is mounted into
the container so the function file is available from the OpenWebUI data path.
