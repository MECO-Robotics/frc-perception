import argparse
from .app import create_app
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='0')
parser.add_argument('--source-2', default=None)
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--detector', choices=['motion','tensorrt'], default=None)
parser.add_argument('--engine', default=None)
args = parser.parse_args()
app = create_app(args.source, args.detector, args.engine, args.source_2)
uvicorn.run(app, host=args.host, port=args.port)
