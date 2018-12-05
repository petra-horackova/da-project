import os
squares_file = 'sources/csv/squares.csv'
seed = 42
train_size = .75
test_size = .25


# corner neighbor weight
corner_neighbor_weight = .5


# spatial parameters
lon_x_min = float(os.environ['lon_x_min'])
lon_x_max = float(os.environ['lon_x_max'])
lat_y_min = float(os.environ['lat_y_min'])
lat_y_max = float(os.environ['lat_y_max'])
city_lon_x_km = 35
city_lat_y_km = 27
