from cv2 import displayOverlay
import plotly.subplots as sp
import plotly.graph_objects as go
from ipywidgets import Dropdown, interact, Output
from IPython.display import display
import math

class Plotting:
    def interactive_plot(self):
            self.fig = go.FigureWidget()
            self.df_dropdown = Dropdown(options=self.trial_data_dict.keys())
            self.df_dropdown.observe(self.update_figure, 'value')
            display(self.df_dropdown)
            self.update_figure(None)
            display(self.fig)

    def update_figure(self, change):
        df_key = self.df_dropdown.value
        num_cols = len(self.trial_data_dict[df_key].columns)
        num_rows = math.ceil(num_cols / 5)
        subplot_titles = tuple(self.trial_data_dict[df_key].columns)
        new_fig = sp.make_subplots(rows=num_rows, cols=5, subplot_titles=subplot_titles)
        for idx, col in enumerate(self.trial_data_dict[df_key].columns, start=1):
            row = math.ceil(idx / 5)
            col_position = idx if idx <= 5 else idx - (5 * (row - 1))
            new_fig.add_trace(go.Scatter(y=self.trial_data_dict[df_key][col], showlegend=False), row=row, col=col_position)
        new_fig.update_layout(height=400*num_rows, width=800, title_text=df_key)
        with self.fig.batch_update():
            self.fig.data = []
            self.fig.layout = new_fig.layout
            for trace in new_fig.data:
                self.fig.add_trace(trace)