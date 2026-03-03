This worker generates a per-job Champ inference config at runtime inside the temp job directory.

The `configs/` directory is kept in the image so you have a stable place for any future custom
templates or overrides, but the current worker does not require a static `champ_inference.yaml`
to exist ahead of time.
