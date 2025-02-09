#!/bin/bash
gunicorn -w 1 -b 0.0.0.0:$PORT 'app:app'
chmod +x start.sh

