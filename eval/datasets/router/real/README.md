# Real photos for the router suite

Drop phone photos / screenshots here; the router suite picks them up as split `test`.
The expected route comes from the file name prefix:

- `drawing__<anything>.jpg` - a technical drawing (photo of a printout, screenshot of a PDF viewer, ...)
- `report_qa__<anything>.jpg` - anything else: a damaged part, a report page, a chart

Or put a sidecar `<file>.json` with `{"expected": "drawing"}` next to the image.
Use only public data (the NIST drawings, the NTSB reports, your own parts) - these files may be committed.
