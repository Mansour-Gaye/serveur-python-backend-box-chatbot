#!/bin/bash
gunicorn -w 1 -k gthread --threads 2 -b 0.0.0.0:$PORT app:app --log-level debug
chmod +x start.sh

