"""Safely verify the configured MongoDB connection without printing credentials."""

from mongo_store import MongoStore


if __name__ == "__main__":
    try:
        MongoStore().initialize()
    except Exception as exc:
        print(f"MongoDB connection failed ({type(exc).__name__}). Check the local connection settings and Atlas Network Access.")
        raise SystemExit(1)
    print("MongoDB connection and application indexes are available.")
