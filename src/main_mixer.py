from src.mixer.app import app
import logging
import uvicorn


def main():
    uvicorn.run(
        app,
        reload=False,
        port=8000,
        log_level=logging.INFO)


if __name__ == "__main__":
    main()
