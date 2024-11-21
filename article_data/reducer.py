#!/bin/python3

import sys
import json

for line in sys.stdin:
    line = line.strip()
    if line:
        try:
            url, content = line.split(':::')
            if 'ksl.com' in url:
                print(json.dumps({'ksl': content}))
            elif 'cnbc.com' in url:
                print(json.dumps({'cnbc': content}))
            elif 'theverge.com' in url:
                print(json.dumps({'the_verge': content}))
            elif 'abcnews.go' in url:
                print(json.dumps({'abc': content}))
            elif 'www.cbsnews.com' in url:
                print(json.dumps({'cbs': content}))
        except ValueError as e:
            print(f"Error processing line: {line} - {e}", file=sys.stderr)
