## Using the Automated Testing Module
Test cases are defined as independent chains of events, one chain per shuttle. Test cases are run automatically on upload and can be stopped at any time. A single error in any shuttle chain will stop the entire test case, this is by design. Check the console in developer tools for more detailed output and error messages.

## Creating Test Cases
Each test case should be in JSON format.

- The JSON object should have a single array called `shuttles`.
- Each object in `shuttles` will have two fields: a string `id`, and an array `events`.
- Each object in `events` will have a string `type`. This can be one of 5 valid shuttle states: `waiting`, `entering`, `looping`, `on_break`, and `exiting`.
- The event object may have additional data depending on the type. For example, `duration` for `on_break` and `waiting`, and `route` for `looping`.

## Additional Notes
- Each shuttle's id must match its internal id representation, a zero-padded 15 digit string. For example, "000000000000001" is shuttle 1.

- `duration` (seconds) must be an integer >= 0.

- `on_break` and `waiting` are treated as equivalent states. This is because currently, the test server converts `on_break` to `waiting` on the next state.

- HTTP errors can very occasionally cause the test case to fail. Rerunning it will usually work.

- There may be subtle timing issues if the test suite tab is unfocused for too long. This is due to background tab throttling, which affects setTimeout and setInterval. This shouldn't affect correctness though.

## Simple Test Case Example
```
{
    "shuttles": [
        {
            "id": "000000000000001",
            "events": [
                {
                    "type": "entering"
                },
                {
                    "type": "on_break",
                    "duration": 30
                },
                {
                    "type": "looping",
                    "route": "NORTH"
                },
                {
                    "type": "exiting"
                }
            ]
        }
    ]
}
```
