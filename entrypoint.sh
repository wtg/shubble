#!/bin/sh

echo "📦 Injecting Vite .env into /app/client"
printenv | grep '^VITE_' > /app/client/.env

echo "⚙️ Building Vite app..."
cd /app/client
npm run build

cd /app
echo "🚀 Starting Flask app..."
exec "$@"
