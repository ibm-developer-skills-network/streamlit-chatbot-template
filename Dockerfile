# Use the base image with Python 3.10, uWSGI, and Nginx
FROM tiangolo/uwsgi-nginx-flask:python3.10

# Create a system group and user for running the application
RUN groupadd -r myuser && useradd -r -g myuser myuser

# Switch to the root user context temporarily (for privileged operations)
USER root 

# Set the working directory inside the Docker container
WORKDIR /streamlit-chatbot-template

# Copy all files from the current build context into the container's working directory
COPY . .

# Install Python packages listed in requirements.txt using pip
RUN pip install -r requirements.txt

# Expose port 8501 for external access
EXPOSE 8501

# Define the entry point command to run when the container starts
ENTRYPOINT ["streamlit", "run"]

# Specify the default Streamlit application script to run
CMD ["💼Portfolio.py"]
