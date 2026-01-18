# Test Files

Test files define sequences of actions for simulated shuttles. When loaded via the test client, each shuttle in the file creates a new simulated shuttle and queues all its actions.

## Format

Test files are JSON with the following structure:

```json
{
  "shuttles": [
    {
      "id": "1",
      "events": [
        { "type": "entering" },
        { "type": "looping", "route": "ROUTE_NAME" },
        { "type": "on_break", "duration": 60 },
        { "type": "exiting" }
      ]
    }
  ]
}
```

## Fields

### Shuttle

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Numeric string identifier for the shuttle (e.g., "1", "2") |
| `events` | array | Yes | Sequence of actions to queue |

### Event

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Action type (see below) |
| `route` | string | For `looping` | Route name to loop on |
| `duration` | number | For `on_break` | Duration in seconds |

## Action Types

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `entering` | Shuttle enters the service area (triggers geofence entry) | - |
| `looping` | Shuttle loops on a route | `route` |
| `on_break` | Shuttle pauses for a duration | `duration` |
| `exiting` | Shuttle exits the service area (triggers geofence exit) | - |

## Available Routes

- `NORTH` - North route
- `WEST` - West route

## Example

A typical shuttle shift:

```json
{
  "shuttles": [
    {
      "id": "1",
      "events": [
        { "type": "entering" },
        { "type": "looping", "route": "NORTH" },
        { "type": "looping", "route": "NORTH" },
        { "type": "on_break", "duration": 300 },
        { "type": "looping", "route": "NORTH" },
        { "type": "exiting" }
      ]
    }
  ]
}
```
