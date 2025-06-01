#!/bin/sh

echo "📦 Environment variables at runtime:"
printenv | sort

echo "🚀 Starting application..."
exec "$@"
