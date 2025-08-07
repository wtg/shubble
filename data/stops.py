import json
import numpy as np

class Stops:
    with open('data/routes.json', 'r') as f:
        routes_data = json.load(f)

    polylines = {}
    for route_name, route in routes_data.items():
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
        closest_point = None
        closest_distance = float('inf')
        closest_route_name = None
        closest_polyline_index = None
        for route_name, polylines in cls.polylines.items():
            for index, polyline in enumerate(polylines):
                # Calculate distances to each segment in the polyline
                # works by finding the closest point on each segment of the polyline
                # using vector projection, then calculating the distance to that point
                # and returning the closest one
                lines = np.array([polyline[:-1], polyline[1:]])
                diffs = lines[1, :] - lines[0, :]
                lengths = np.linalg.norm(diffs, axis=1)
                diffs_normalized = diffs / lengths[:, np.newaxis]
                projections = np.sum((point - lines[0, :]) * diffs_normalized, axis=1)
                projections = np.clip(projections, 0, lengths)
                closest_points = lines[0, :] + projections[:, np.newaxis] * diffs_normalized
                distances = np.linalg.norm(point - closest_points, axis=1)
                min_index = np.argmin(distances)
                if distances[min_index] < closest_distance:
                    closest_distance = distances[min_index]
                    closest_point = closest_points[min_index]
                    closest_route_name = route_name
                    closest_polyline_index = index
        return closest_point, closest_distance, closest_route_name, closest_polyline_index

    @classmethod
    def is_at_stop(cls, origin_point, threshold=0.0002):
        """
        Check if the given point is close enough to any stop.
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :param threshold: Distance threshold to consider as "at stop".
        :return: the stop name if close enough, otherwise None.
        """
        for route_name, route in cls.routes_data.items():
            for stop in route.get('STOPS', []):
                stop_point = np.array(route[stop]['COORDINATES'])
                distance = np.linalg.norm(np.array(origin_point) - stop_point)
                if distance < threshold:
                    return route_name, stop
        return None, None
