# Use Python 3.11-slim as the base image
FROM python:3.11.4

# Set the default working directory
WORKDIR /app

# Copy everything from the current directory to the container's root
COPY . /app

# Install required system packages (for geopandas)
RUN apt-get update && apt-get install -y libgeos-dev

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
