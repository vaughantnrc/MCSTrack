import logging
import uvicorn


def main():
    uvicorn.run(
        "calibrator.calibrator_app:app",
        reload=False,
        port=7999,
        log_level=logging.INFO)


if __name__ == "__main__":
    main()
