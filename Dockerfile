FROM python:3.12-slim

# Install git and other dependencies
RUN apt-get update && apt-get install -y git swig g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a directory for data sharing
RUN mkdir -p /app/recordings
RUN mkdir -p /app/agents

# clone and install repo
ARG REPO_URL="https://github.com/fhstp/MLPragueImitation"
RUN git clone ${REPO_URL} repo
RUN pip install --upgrade pip
RUN pip install ./repo
RUN cp repo/backend.py .

# change to headless version of cv2
RUN pip uninstall -y opencv-python
RUN pip install opencv-python-headless

# Expose the port
EXPOSE 8000

# Volume for recordings
VOLUME ["/app/recordings"]
VOLUME ["/app/agents"]

# create file to run backend
RUN echo '#!/bin/bash\ncd /app\npython /app/backend.py' > /app/run.sh && chmod +x /app/run.sh

# Command to run the server
ENTRYPOINT ["/app/run.sh"]
