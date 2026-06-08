#!/bin/bash
# Setup script for Streamlit Cloud deployment
mkdir -p ~/.streamlit/

cat > ~/.streamlit/config.toml <<EOF
[server]
headless = true
port = $PORT
enableCORS = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
EOF
