import uvicorn
import argparse
from src.api.main import create_app


def main():
    parser = argparse.ArgumentParser(description="LocalSearch API Server")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()
    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
