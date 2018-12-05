import time

import geopandas as gp
import sklearn.model_selection as sms
from sklearn.metrics import r2_score, mean_squared_error

import cellmodel as cm
import constants as cons
import datareader as dr
import geospatial as geo
import plotter as plot
import pandas as pd

start_time = int(round(time.time()))

# %% suffix of output directory name
CUSTOM_PARAMS = ""

# %% data pre-processing
my_cube = dr.get_df_cube(24, 27, 35, cons.squares_file)

features, target, hours = cm.get_features_target(my_cube)
features.loc[:, 'h'] = hours

# train test split
X_train, X_test, y_train, y_test = sms.train_test_split(features, target,
                                                        train_size=cons.train_size, test_size=cons.test_size,
                                                        random_state=cons.seed)
# %% fit model on train
my_model = cm.Model()
my_model.fit(X_train, y_train)

# predict
pred_y_train = my_model.predict(X_train)
pred_y_test = my_model.predict(X_test)

print('R2 (test): ' + str(round(r2_score(y_test, pred_y_test), 4)))
print('R2 (train): ' + str(round(r2_score(y_train, pred_y_train), 4)))
print('MSE (test): ' + str(round(mean_squared_error(y_test, pred_y_test), 4)))
print('MSE (train): ' + str(round(mean_squared_error(y_train, pred_y_train), 4)))

# export
X_train.loc[:, 'prediction'] = pred_y_train.round(2)
X_train.loc[:, 'target'] = y_train.round(2)

X_test.loc[:, 'prediction'] = pred_y_test.round(2)
X_test.loc[:, 'target'] = y_test.round(2)

scored = X_train.append(X_test)
scored = scored.sort_index()

for sim_hour in range(my_cube.shape[0] - 23):
    sim_hour = 0
    local_df = scored
    print("running simulation for h: ", sim_hour)
    my_sim = cm.Simulation(initial_state=my_cube[sim_hour], starting_hour=sim_hour, model=my_model)
    my_sim.run()
    sim_feat, sim_tar, sim_hrs = cm.get_features_target(my_sim._data)
    local_df.loc[:, 'simulation'] = sim_tar

    # %%
    print('preparing plots for simulation with starting hour', sim_hour)
    directory_name_details = CUSTOM_PARAMS + "_h_" + str(sim_hour)
    output_directory = dr.prepare_output_dir(directory_name_details)
    sim_geo = geo.create_geo_df(local_df, 'lon_x', 'lat_y')

    # %% last cell
    grid = gp.GeoDataFrame.from_file('sources/shapefiles/grid_1x1_km.shp')
    grid.drop(columns=['xmin', 'xmax', 'ymin', 'ymax'])

    complete_dfs = []
    for h in sim_geo.h.unique():
        plot_title = 'simulation starting hour: ' + str(sim_hour) + '\n current h = ' + str(h)
        plotter = plot.CustomPlotter(hour=h, output_dir=output_directory, title=plot_title, max_rows=2, max_cols=3)

        print('plotting hour: ', h)
        current = sim_geo.loc[sim_geo['h'] == h]
        initial_join = gp.sjoin(current, grid, how='inner', op='within')
        cut_join = initial_join.drop(columns=['geometry'])  # drops point geometry column from joined gdf
        cut_grid = grid.drop(columns=['xmin', 'xmax', 'ymin', 'ymax'])  # leaves only id and geometry from grid
        square_test = cut_join.join(cut_grid.set_index('id'),
                                    on='index_right')  # joins the grid geometry from cut_grid to proper index in joined gdf

        complete_dfs.append(square_test)

        # current_value plot
        plotter.plot_gdf_column(square_test, 'current_value', 'current value (h)', row_pos=0, col_pos=0)

        # prepare and plot details table
        r2_pred = round(r2_score(square_test.loc[:, 'target'], square_test.loc[:, 'prediction']), 4)
        r2_sim = round(r2_score(square_test.loc[:, 'target'], square_test.loc[:, 'simulation']), 4)
        mse_pred = round(mean_squared_error(square_test.loc[:, 'target'], square_test.loc[:, 'prediction']), 4)
        mse_sim = round(mean_squared_error(square_test.loc[:, 'target'], square_test.loc[:, 'simulation']), 4)

        table_data = (
            {
                'prediction': [
                    r2_pred,
                    mse_pred],
                'simulation': [
                    r2_sim,
                    mse_sim]
            }
        )

        table = pd.DataFrame(data=table_data)
        table = table.rename({0: '   R2   ', 1: '   MSE   '}, axis='index')

        plotter.plot_table(table, row_pos=0, col_pos=1, colspan=2)
        plotter.plot_gdf_column(square_test, 'target', 'target (h+1)', row_pos=1, col_pos=0, cmap='Greens')
        plotter.plot_gdf_column(square_test, 'prediction', 'prediction (h+1)', row_pos=1, col_pos=1, cmap='Oranges')
        plotter.plot_gdf_column(square_test, 'simulation', 'simulation (h+1)', row_pos=1, col_pos=2, cmap='RdPu')
        plotter.save_figs()
        plotter.show_and_close()

    plotter.make_gif()

    final_df_out = complete_dfs[0]
    for i in range(1, len(complete_dfs)):
        final_df_out = final_df_out.append(complete_dfs[i])

    final_df_out.to_csv(output_directory + '/full_sim.csv')


end_time = int(round(time.time()))
print('elapsed time [s]: ', end_time - start_time)
