CST8504 Assignment1_Prediction_Dashboard


# Build the image from the Dockerfile in the current folder (.)
# -t gives the image a name (tag)
docker build -t team_name_dashboard .
 
# Run a container from the image
# -p 8501:8501  maps port 8501 on your computer to port 8501 in the container
# --name        gives the container a readable name
# --rm          removes the container automatically when it stops
docker run --rm -p 8501:8501 --name dashboard team_name_dashboard
