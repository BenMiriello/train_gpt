# Use the official PyTorch image as the base
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Install necessary dependencies
RUN pip install transformers
RUN pip install pandas
RUN pip install accelerate

# Copy the script files into the container
COPY train_model.py /app/
COPY generate_responses.py /app/
COPY dataset.csv /app/
COPY test_model.py /app/

# Run the training script when the container starts
CMD ["python", "train_model.py"]
