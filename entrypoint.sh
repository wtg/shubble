#!/bin/sh

echo "📦 Injecting .env into /app/client"

# Create .env file for Vite with only VITE_ prefixed vars
printenv | grep '^VITE_' > /app/client/.env

echo "✅ /app/client/.env contents:"
cat /app/client/.env

echo "🚀 Starting application: $@"
exec "$@"
