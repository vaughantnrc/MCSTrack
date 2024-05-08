import logging
import uvicorn


def main():
    uvicorn.run(
        "detector.detector_app:app",
        reload=False,
        port=8001,
        log_level=logging.INFO)


if __name__ == "__main__":
    main()
