#!/bin/sh
set -e

# Substitute environment variables into config.json
envsubst < /usr/share/nginx/html/config.template.json > /usr/share/nginx/html/config.json

# Start nginx
exec nginx -g 'daemon off;'
