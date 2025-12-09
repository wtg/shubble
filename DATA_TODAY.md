# data_today()

### Reference

Regarding the `data_today()` function in path `C:\...\shubble\server\routes.py`.

### Outputted JSON Format:

```bash
{
    "shuttleID1": {
        "locations": {
            "16:25:53": {
                "at_stop": null,
                "closest_polyline": 1,
                "closest_route": "NORTH",
                "closest_route_location": [42.7379143365656, -73.6675641148277],
                "distance": 0.0100496703554305,
                "latitude": 42.7380042100651,
                "longitude": -73.6675511200586
            },
            "16:26:01": { ... },
            "16:26:08": { ... }
            // ...
        },
        "breaks": [
            {
                "locations": [
                    "16:25:53",     // break start time
                    "16:26:01",
                    "16:26:08",
                    "16:26:15",
                    "16:26:22"      // break end time
                ]
            },
            {
                "locations": [ ... ]
            }
        ],
        "loops": [
            {
                "locations": [
                    "16:28:08",     // loop start time
                    "16:28:16",
                    "16:28:23",
                    "16:28:30",
                    "16:28:37"      // loop end time
                ]
            },
            {
                "locations": [ ... ]
            }
        ]
    },
    "shuttleID2": { ... },
    "shuttleID3": { ... }
    // ...
``` 