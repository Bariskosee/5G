"""Main CLI entry point — called by Docker CMD.

Reads a single video file from /app/data/input/ (or a path supplied via
--input), runs the full hybrid AI pipeline, and writes results.json to
/app/data/output/ (or the path supplied via --output).

Responsibilities:
  - Parse CLI arguments (--input, --output, --config).
  - Validate that the input file exists.
  - Instantiate and execute src.pipeline.Pipeline.
  - Exit with code 0 on success, 1 on any error.
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "src/predict.py is a placeholder. Implement the CLI entry point before running."
    )
