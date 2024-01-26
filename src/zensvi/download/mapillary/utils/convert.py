from shapely.geometry import shape, mapping, MultiPolygon, MultiLineString, MultiPoint

def convert_multi_feature_collection(feature_collection):
    """
    Convert all MultiPolygon, MultiLineString, or MultiPoint features in a GeoJSON FeatureCollection 
    to individual Polygon, LineString, or Point features, respectively.
    
    :param feature_collection: A GeoJSON FeatureCollection.
    :return: A new GeoJSON FeatureCollection with converted features.
    """
    converted_features = []

    for feature in feature_collection['features']:
        geometry_type = feature['geometry']['type']
        shapely_geometry = shape(feature['geometry'])

        if isinstance(shapely_geometry, (MultiPolygon, MultiLineString, MultiPoint)):
            for geom in shapely_geometry.geoms:
                singular_feature = {
                    'type': 'Feature',
                    'properties': feature['properties'],  # Copy the properties
                    'geometry': mapping(geom)
                }
                converted_features.append(singular_feature)
        else:
            converted_features.append(feature)  # If not a Multi type, return as is

    return {'type': 'FeatureCollection', 'features': converted_features}


def extract_coordinates_from_polygons(feature_collection):
    """
    Check if all features in the GeoJSON are Polygons or MultiPolygons.
    If so, extract their coordinates as a list of lists (each list contains tuples of longitude, latitude).
    Raise an error if any feature is not a Polygon or MultiPolygon.

    :param feature_collection: A GeoJSON FeatureCollection.
    :return: A list of lists with coordinates.
    """
    polygons_coordinates = []

    for feature in feature_collection['features']:
        geometry_type = feature['geometry']['type']
        shapely_geometry = shape(feature['geometry'])

        if geometry_type not in ['Polygon', 'MultiPolygon']:
            raise ValueError("All features must be Polygons or MultiPolygons. Found: " + geometry_type)

        if isinstance(shapely_geometry, MultiPolygon):
            for polygon in shapely_geometry.geoms:
                # Append each Polygon's exterior coordinates as a new list
                polygons_coordinates.append(list(polygon.exterior.coords))
        elif geometry_type == 'Polygon':  # It's a Polygon
            # Append the Polygon's exterior coordinates as a new list
            polygons_coordinates.append(list(shapely_geometry.exterior.coords))

    return polygons_coordinates

# Example usage:
# coordinates = extract_coordinates_from_polygons(your_feature_collection)
