FROM nikolaik/python-nodejs:python3.13-nodejs24

WORKDIR /app

# Install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server app and React build output
COPY server/ ./server/
COPY client/ ./client/

# Install Node.js dependencies and build vite app
WORKDIR /app/client
RUN npm install

# move back to app
WORKDIR /app

# Set environment variable for Flask (production mode)
ENV FLASK_DEBUG=false
ENV FLASK_ENV=production
ENV FLASK_HOST=0.0.0.0

# Expose port
EXPOSE 80

# run setup script
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

# Start Flask app
CMD ["python", "server/shubble.py"]
