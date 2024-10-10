FROM python:3.9

# Expose the container
EXPOSE 8080

# Get all packages needed within the docker container
RUN pip install -U pip
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy the app within the container and change to the app workdir
COPY . /app
WORKDIR /app

# Using entrypoiny as we are using the container as en exec
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
