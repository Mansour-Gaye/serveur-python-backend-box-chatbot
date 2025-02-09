#!/bin/bash
gunicorn -w 1 'app:app'
chmod +x start.sh

