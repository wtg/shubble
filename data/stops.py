import json

class NorthStops:

    with open('data/north_route.json', 'r') as f:
        north_stops_data = json.load(f)

    STUDENT_UNION = north_stops_data['STUDENT_UNION']
    COLONIE = north_stops_data['COLONIE']
    STAC_1 = north_stops_data['STAC_1']
    STAC_2 = north_stops_data['STAC_2']
    STAC_3 = north_stops_data['STAC_3']
    ECAV = north_stops_data['ECAV']
    HOUSTON_FIELD_HOUSE = north_stops_data['HOUSTON_FIELD_HOUSE']

    ROUTES = north_stops_data['ROUTES']

class WestStops:

    with open('data/west_route.json', 'r') as f:
        west_stops_data = json.load(f)

    STUDENT_UNION = west_stops_data['STUDENT_UNION']
    ACADEMY_HALL = west_stops_data['ACADEMY_HALL']
    POLYTECHNIC = west_stops_data['POLYTECHNIC']
    CITY_STATION = west_stops_data['CITY_STATION']
    BLITMAN = west_stops_data['BLITMAN']
    WEST_HALL = west_stops_data['WEST_HALL']
    _86_FIELD = west_stops_data['_86_FIELD']

    ROUTES = west_stops_data['ROUTES']
