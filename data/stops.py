import json
import numpy as np

class Stops:
    with open('data/routes.json', 'r') as f:
        routes_data = json.load(f)

    with open('data/schedule.json', 'r') as f:
        schedule_data = json.load(f)

    # get active routes from schedule
    active_routes = set()
    for day in ['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY']:
        if day in schedule_data:
            schedule_name = schedule_data[day]
            for bus_schedule in schedule_data[schedule_name].values():
                for time, route_name in bus_schedule:
                    active_routes.add(route_name)

    polylines = {}
    for route_name in active_routes:
        route = routes_data.get(route_name)
        polylines[route_name] = []
        for polyline in route.get('ROUTES', []):
            polylines[route_name].append(np.array(polyline))

    @classmethod
    def get_closest_point(cls, origin_point):
        """
        Find the closest point on any polyline to the given origin point.
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :return: A tuple with the closest point (latitude, longitude), distance to that point,
                route name, and polyline index.
        """
        point = np.array(origin_point)

        closest_data = []
        for route_name, polylines in cls.polylines.items():
            for index, polyline in enumerate(polylines):
                if len(polyline) < 2:
                    # not enough points to form a segment, just check distance to the point itself
                    closest_data.append((np.linalg.norm(point - np.array(polyline[0])), np.array(polyline[0]), route_name, 0))
                    continue

                # Build segments
                lines = np.array([polyline[:-1], polyline[1:]])
                diffs = lines[1, :] - lines[0, :]
                lengths = np.linalg.norm(diffs, axis=1)

                # Handle zero-length segments (duplicate points)
                nonzero_mask = lengths > 0
                if np.any(~nonzero_mask):
                    # check distance directly to these points
                    zero_points = lines[0, ~nonzero_mask]
                    zero_distances = np.linalg.norm(point - zero_points, axis=1)
                    min_idx = np.argmin(zero_distances)
                    closest_data.append((zero_distances[min_idx], zero_points[min_idx], route_name, min_idx))

                if not np.any(nonzero_mask):
                    # all segments are zero-length, already handled
                    continue

                diffs_normalized = diffs[nonzero_mask] / lengths[nonzero_mask, np.newaxis]
                projections = np.sum((point - lines[0, nonzero_mask]) * diffs_normalized, axis=1)
                projections = np.clip(projections, 0, lengths[nonzero_mask])
                closest_points = lines[0, nonzero_mask] + projections[:, np.newaxis] * diffs_normalized
                distances = np.linalg.norm(point - closest_points, axis=1)

                min_index = np.argmin(distances)
                closest_data.append((distances[min_index], closest_points[min_index], route_name, index))

        # Find the overall closest point
        if closest_data:
            closest_routes = sorted(closest_data, key=lambda x: x[0])
            # Check if closest route is significantly closer than others
            if len(closest_routes) > 1 and closest_routes[1][0] - closest_routes[0][0] < 0.0001:
                # If not significantly closer, return None to indicate ambiguity
                return None, None, None, None
            return closest_routes[0]
        return None, None, None, None

    @classmethod
    def is_at_stop(cls, origin_point, threshold=0.0002):
        """
        Check if the given point is close enough to any stop.
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :param threshold: Distance threshold to consider as "at stop".
        :return: A tuple with (the route name if close enough, otherwise None,
                the stop name if close enough, otherwise None).
        """
        for route_name, route in cls.routes_data.items():
            for stop in route.get('STOPS', []):
                stop_point = np.array(route[stop]['COORDINATES'])
                distance = np.linalg.norm(np.array(origin_point) - stop_point)
                if distance < threshold:
                    return route_name, stop
        return None, None
