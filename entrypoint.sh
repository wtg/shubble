#!/bin/sh

# inject .env into vite app
set -e
printenv | grep '^VITE_' > /app/client/.env

# build vite app
cd /app/client
npm run build

# remove .env file
rm /app/client/.env

# run CMD and subsequently the flask app
exec "$@"
