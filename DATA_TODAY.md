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

### Future Work

Regarding 
* _Priority:_ To fully close Issue #59, Domain Staging needs to be completed.

Regarding the smaller tweaks and changes:
* Tweak the thresholds, listed at the top of the `data_today()` function, via testing (on the test-server and staging domain). Keep in-mind that some of these thresholds may also be used across different files, meaning it may be better to update it everywhere and/or implement a universally shared value. 
* Determining whether or not the shuttle is stopped (boolean `is_stopped`) is calculated by subtracting latitude and longitude. This improper method of subtraction is done in other files, so a comprehensive fix across everything would be ideal. 