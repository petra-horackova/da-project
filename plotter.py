import os

import geopandas as gp
import imageio
import matplotlib.colors as mcols
import matplotlib.pyplot as plto


class CustomPlotter:
    def __init__(self, hour, output_dir, title, max_rows=2, max_cols=3):
        self._plt = plto
        self._output_dir = output_dir
        self._plt.rcParams.update({'font.size': 4})
        self._plt.tight_layout()
        self._plt.suptitle(title, fontsize=8)
        self._plt.subplots_adjust(top=1, wspace=0, hspace=0)
        self._plt.interactive(True)

        self._h = hour
        self._axes = self._plt.gca()
        self._axes.autoscale()
        self._max_rows = max_rows
        self._max_cols = max_cols

    def plot_table(self, table, row_pos, col_pos, colspan):
        ax = self._plt.subplot2grid((self._max_rows, self._max_cols), (row_pos, col_pos), colspan=colspan)
        the_table = ax.table(cellText=table.values, colWidths=[0.08] * len(table.columns), colLabels=table.columns,
                             rowLabels=table.index, loc='center', cellLoc='center')
        the_table.scale(2, 1.35)
        the_table.set_fontsize(4)
        ax.axis('off')

    def plot_gdf_column(self, geo_df, column_name, title, row_pos, col_pos, cmap='Blues'):
        ax = self._plt.subplot2grid((self._max_rows, self._max_cols), (row_pos, col_pos))
        ax.set_title(title)
        ax.axis('off')
        min_value = geo_df.loc[geo_df[column_name].idxmin()].loc[column_name]
        max_value = geo_df.loc[geo_df[column_name].idxmax()].loc[column_name]
        city_gdf = gp.GeoDataFrame.from_file('sources/shapefiles/city.shp')
        city_gdf.plot(ax=ax, linewidth=0.4, color='white', edgecolor='black')

        my_cmap = self._plt.cm.get_cmap(cmap)
        anom_norm = mcols.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=min_value, vmax=max_value)
        geo_df.plot(column=column_name, ax=ax, cmap=my_cmap, linewidth=0.05,
                    edgecolor='0.05', norm=anom_norm, vmin=0.1, alpha=0.8)

    def save_figs(self):
        # podminka zajistujici ze ploty s cislem min nez 10 se ulozi jako 00x aby bylo zajisteny spravny poradi souboru
        # jinak by system automaticky radil ploty jako plot_1, plot_10, plot_11... plot_2, plot_20 atd
        if self._h < 10:
            self._plt.savefig(self._output_dir + '/plot_00' + str(self._h) + '.png', dpi=400)
        else:
            self._plt.savefig(self._output_dir + '/plot_0' + str(self._h) + '.png', dpi=400)

    def show_and_close(self):
        self._plt.show()
        self._plt.close()

    def make_gif(self):
        print('exporting gif file')
        images = []
        for file_name in os.listdir(self._output_dir):
            if file_name.endswith('.png'):
                file_path = os.path.join(self._output_dir, file_name)
                images.append(imageio.imread(file_path))
        imageio.mimsave(self._output_dir + '/plots_auto.gif', images, duration=0.8)







