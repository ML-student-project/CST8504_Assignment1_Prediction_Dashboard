
# 1. Start from an official, lightweight Python image
FROM python:3.10-slim


# 2. Set the working folder inside the container
WORKDIR /app

# 3. Copy only requirements first, then install them.
#    Docker caches this layer, so libraries are not reinstalled
#    every time you change your code.
COPY requirements.txt .

# Remove 'tensorflow_cpu' before installing as we are using Decision Tree for the final model.
# Note: docker image with tensor-cpu installed = 3Gb
#       docker image without tensorflow_cpu installed = 1.39G
RUN sed '/^tensorflow_cpu/d' requirements.txt > filtered_requirements.txt && \
    pip install --no-cache-dir -r filtered_requirements.txt

# 4. Copy the rest of the project (code, models, processed data)
COPY . .

# 5. Document the port Streamlit listens on
EXPOSE 8501

# 6. Command that runs when the container starts.
#    0.0.0.0 makes the app reachable from outside the container.
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
