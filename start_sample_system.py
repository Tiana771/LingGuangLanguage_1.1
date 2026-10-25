"""Start this workspace on port 5001 with v2 semantic knowledge search."""
from serve_sign_api import main


if __name__ == "__main__":
    raise SystemExit(main(["--semantic-books", "--port", "5001"]))
