from cv2 import displayOverlay
import pandas as pd
import os
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
from ipywidgets import Dropdown, interact, Output
from IPython.display import display
import math
import pickle



class CalciMetrics:
    def __init__(self, path_to_data):
        self.trial_data_dict={}
        self.path_to_data=path_to_data

    def import_data(self, first_col_frames=True, frames_to_seconds=True, fps_rate=1, background_sub=True, background_col=-1):
        file_names=[f for f in os.listdir(self.path_to_data) if f.endswith('.csv')]
        list_of_paths=[os.path.join(self.path_to_data, f) for f in file_names]
        
        for i, ii in zip(list_of_paths, file_names):
            import_df=pd.read_csv(i)
            if first_col_frames:
                import_df.set_index(import_df.columns[0], inplace=True)
                import_df.drop(columns=import_df.columns[0], inplace=True)
            
            if frames_to_seconds:
                import_df.index=import_df.index/fps_rate
                self.x_label='Time (s)'
            else:
                self.x_label='Frames'

            if background_sub:
                import_df=import_df.sub(import_df[background_col])
                import_df.drop(columns=import_df.columns[background_col])

            
            self.trial_data_dict[ii]=import_df
        
    def dff_calc(self, df, method, percentile=0.2, first_frame=None, last_frame=None ):
        if method=='median':
            f0=df.median()
        elif method=='percentile':
            f0=df.quantile(percentile)
        elif method=='first_frame':
            f0=df.iloc[0,:]
        elif method=='multi_frame':
            f0=df.iloc[first_frame:last_frame, :]

        df=(df-f0)/f0

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

    def exclude_rois(self, excluded_rois_dict):

        for i in excluded_rois_dict:
            cleaned_df=self.trial_data_dict[i].copy()
            exclude_columns=cleaned_df[excluded_rois_dict[i]].columns
            cleaned_df=cleaned_df.drop(columns=exclude_columns)    
            self.trial_data_dict[i]=cleaned_df
          
    def get_summary_statistics(self):
        self.summary_dict={}
        for i in self.trial_data_dict:
            value_dict={}
            value_dict['mean']=self.trial_data_dict[i].mean(axis=1)
            value_dict['median']=self.trial_data_dict[i].median(axis=1)
            value_dict['std']=self.trial_data_dict[i].std(axis=1)
            value_dict['sem']=self.trial_data_dict[i].sem(axis=1)
            self.summary_dict[i]=value_dict
        
            
    def save(self, filename, save_trials_to_csv=False):
        full_path=os.path.join(self.path_to_data, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(self, f)
        if save_trials_to_csv:
            save_path='/'.join([self.path_to_data,'results'])
            os.mkdir(save_path)
            for i in self.trial_data_dict:
                self.trial_data_dict[i].to_csv(''.join([save_path,'/',i,'.csv']))


    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

