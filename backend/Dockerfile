# Use an official Python runtime as the base image
FROM python:3.11.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install libgl1-mesa-glx
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the application code to the container
COPY . .

# Expose the port that the Flask server will be listening on
EXPOSE 5000

# Set the entrypoint command to run the Flask server
CMD ["python", "app.py"]