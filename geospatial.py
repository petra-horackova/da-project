import constants as cons
import geopandas as gp
from shapely.geometry import Point

def create_geo_df(dataframe, column_i,column_j):
    # i, j series, treba scored['lon_x'], scored['lon_x'], scored je dataframe
    lon_x_geo = cons.lon_x_min + (dataframe[column_i] + 0.5) * ((cons.lon_x_max - cons.lon_x_min) / cons.city_lon_x_km)
    lat_y_geo = cons.lat_y_min + (dataframe[column_j] + 0.5) * ((cons.lat_y_max - cons.lat_y_min) / cons.city_lat_y_km)
    dataframe.loc[:, 'lon_x_geo'] = lon_x_geo
    dataframe.loc[:, 'lat_y_geo'] = lat_y_geo
    # combine lat and lon column to a shapely Point() object
    dataframe['geometry'] = dataframe.apply(lambda x: Point((float(x.lon_x_geo), float(x.lat_y_geo))), axis=1)
    dataframe_geo = gp.GeoDataFrame(dataframe, geometry='geometry')
    # add coordinate reference system, proj WGS84
    dataframe_geo.crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    return dataframe_geo
