import json
import numpy as np
import math

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
                    zero_distances = haversine_vectorized(point[np.newaxis, :], zero_points)
                    min_idx = np.argmin(zero_distances)
                    closest_data.append((zero_distances[min_idx], zero_points[min_idx], route_name, min_idx))

                if not np.any(nonzero_mask):
                    # all segments are zero-length, already handled
                    continue

                diffs_normalized = diffs[nonzero_mask] / lengths[nonzero_mask, np.newaxis]
                projections = np.sum((point - lines[0, nonzero_mask]) * diffs_normalized, axis=1)
                projections = np.clip(projections, 0, lengths[nonzero_mask])
                closest_points = lines[0, nonzero_mask] + projections[:, np.newaxis] * diffs_normalized
                distances = haversine_vectorized(point[np.newaxis, :], closest_points)

                min_index = np.argmin(distances)
                closest_data.append((distances[min_index], closest_points[min_index], route_name, index))

        # Find the overall closest point
        if closest_data:
            closest_routes = sorted(closest_data, key=lambda x: x[0])
            # Check if closest route is significantly closer than others
            if len(closest_routes) > 1 and haversine(closest_routes[0][1], closest_routes[1][1]) < 0.020:
                # If not significantly closer, return None to indicate ambiguity
                return None, None, None, None
            return closest_routes[0]
        return None, None, None, None

    @classmethod
    def is_at_stop(cls, origin_point, threshold=0.020):
        """
        Check if the given point is close enough to any stop.
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :param threshold: Distance threshold to consider as "at stop".
        :return: A tuple with (the route name if close enough, otherwise None,
                the stop name if close enough, otherwise None).
        """
        origin = np.array(origin_point).reshape(1,2)

        for route_name, route in cls.routes_data.items():
            for stop in route.get('STOPS', []):
                stop_point = route.get(stop)
                if not stop_point or "COORDINATES" not in stop_point:
                    continue
                
                stop_obj = np.array(stop_point["COORDINATES"]).reshape(1,2)
                stop_name = stop_point.get("NAME", stop)

                distance = haversine_vectorized(origin, stop_obj)[0]
                if distance < threshold:
                    return route_name, stop_name

        return None, None

def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points on the Earth
    using the Haversine formula.

    Parameters:
        coord1: (lat1, lon1) in decimal degrees
        coord2: (lat2, lon2) in decimal degrees

    Returns:
        Distance in kilometers.
    """
    # Earth radius in kilometers
    R = 6371.0

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def haversine_vectorized(coords1, coords2):
    """
    Vectorized haversine distance between two sets of coordinates.

    Parameters
    ----------
    coords1 : array_like, shape (N, 2)
        Array of (lat, lon) pairs in decimal degrees.
    coords2 : array_like, shape (N, 2)
        Array of (lat, lon) pairs in decimal degrees.

    Returns
    -------
    distances : ndarray, shape (N,)
        Great-circle distances in kilometers.
    """
    # Accept either single (lat,lon) pairs or arrays of pairs. Normalize to 2-D arrays.
    coords1 = np.atleast_2d(np.asarray(coords1, dtype=float))
    coords2 = np.atleast_2d(np.asarray(coords2, dtype=float))

    # Earth radius in kilometers
    R = 6371.0

    lat1 = np.radians(coords1[:, 0])
    lon1 = np.radians(coords1[:, 1])
    lat2 = np.radians(coords2[:, 0])
    lon2 = np.radians(coords2[:, 1])

    dphi = lat2 - lat1
    dlambda = lon2 - lon1

    a = np.sin(dphi / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
