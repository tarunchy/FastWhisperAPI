#!/bin/bash

# Define the host, port, and log file
HOST="0.0.0.0"
PORT=8001  # Change this to the port you want
LOG_FILE="fastapi.log"

# Run the FastAPI app in the background using uvicorn
nohup uvicorn main:app --host $HOST --port $PORT --reload > $LOG_FILE 2>&1 &
echo "FastAPI is running in the background on port $PORT."
