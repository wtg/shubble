# Setup

### Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher
- PostgreSQL 17 or higher

You can install these using your system's package manager or download them from their official websites. When installing PostgreSQL, make sure to disable password authentication for localhost connections.

### Clone the Repo
```bash
git clone git@github.com:wtg/shubble.git
cd shubble
```

### Install dependencies
```bash
# Python dependencies
pip install -r requirements.txt
# Node.js dependencies
npm install
```
Note: you may want to install Python requirements in a virtual environment to avoid conflicts with other projects.

### Setup Database
Create a new database named `shubble`:
```bash
createdb shubble
```
Initialize the database:
```bash
flask db init
```

### Verify Database Setup
To verify that the database is set up correctly, you can run the following command to open a PostgreSQL interactive terminal:
```bash
psql shubble
```
This will connect you to the `shubble` database. You can then run SQL commands to check the tables and data. For example, to list all tables:
```sql
\dt
```
You should see the tables `vehicles`, `geofence_events`, and `vehicle_locations` listed.

# Running the frontend
To run the frontend, `cd` to the project root and run:
```bash
npm run dev
```
This will start the development server and open the frontend in your default web browser. The frontend will automatically reload when you make changes to the source files.
Note: `npm run dev` is for development only. It serves dynamic files and will not work with the backend. You should only use `npm run dev` when you are developing a purely frontend change.

To build the frontend for the backend to use, run:
```bash
npm run build
```
This will create a static build of the frontend in the `/client/dist` directory, which the backend can serve. **You must build the frontend before you run the backend**.

# Running the backend
To run the backend, you need to run the _server_ and the _worker_. They must be running simultaneously for Shubble to work correctly. This means you may have to make 2 terminal tabs.

#### To run the backend server, you have 2 options:

#### Option 1. `cd` to the project root and run:
```bash
flask run --port 8000
```
This will start the Flask development server on port 8000. The backend will serve the built frontend files from the `/client/dist` directory.

#### Option 2. Run the backend using `gunicorn`, which is what Shubble's production server runs:
```bash
gunicorn shubble:app
```

#### To run the worker, `cd` to the project root and run:
```bash
python -m server.worker
```
This will start the worker process that handles background tasks, such as updating vehicle locations. It's important that you run it using `python -m server.worker` (as a python package) so that it can find its local imports.

# Testing the backend
To test the backend, Shubble provides another Flask app that mimics the Samsara API. The test app enables users to trigger shuttle entry, exit, and location updates without needing to set up a real Samsara account or API keys. This is useful for development and testing purposes.
**Note**: even if you're not developing the backend, you may still want to run the test to populate Shubble with data.
Like Shubble, the test app is built using Flask and React. Therefore, you must build the frontend before running the test app.
To build the frontend for the test app, `cd` to the `/test-client` directory and run:
```bash
npm run build
```
This will create a static build of the test app in the `/test-client/dist` directory.
Then, you can run the test app using Flask. From the project root:
```bash
python -m test-server.server
```
This will start a server at `localhost://8001`, which simulates what the Samsara API server would be doing for Shubble in production. It sends simulated webhook requests for vehicles entering/exiting the RPI geofence and sends simulated shuttle location data.
It expects the Shubble server to be running on `localhost://8000`, so make sure to start the backend server first.

# Database Migrations

Most of the time, you will not need to worry about the database schema. However, if you do need to make changes, you can use Flask-Migrate to handle database migrations.

Some background: PostgreSQL is a _database management system_, a system for creating, managing, and querying databases.
PostgreSQL (often abbreviated as Postgres) is a powerful, open-source _object-relational_ database system.
Object-relational means that it stores data in a tabular format.

Some organization and terminology:

```
database: shubble (you create this)
    |
    schema: public (default schema, don't worry about this for now)
        |
        tables: vehicles, geofence_events, vehicle_locations
            |
            rows: representing shuttles (vehicles), entry/exit events (geofence_events), and instances of vehicle location data (vehicle_locations)
                |
                columns: attributes, such as shuttle ids, event types, timestamps, etc.
```


When you need to make changes to the database schema, you can use Flask-Migrate to handle database migrations. This allows you to version control your database schema and apply changes incrementally.

To create a new migration, modify `models.py` to reflect your change and then run:
```bash
flask db migrate -m "Add new attribute"
```
Then apply the migration using:
```bash
flask db upgrade
```
Migration files will be generated in the `migrations` directory. You should commit these files to the Git repository so that any database changes you make can be mirrored by others using `upgrade`.
