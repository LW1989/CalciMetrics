import pandas as pd
import os
import numpy as np
import pickle


class CalciMetrics:
    def __init__(self, path_to_data):
        self.trial_data_dict={}
        self.path_to_data=path_to_data
        #self.plotter=Plotting(self)

    def import_data(self, first_col_frames=True, frames_to_seconds=True, fps_rate=1, background_sub=True, background_col=-1):
        file_names=[f for f in os.listdir(self.path_to_data) if f.endswith('.csv')]
        list_of_paths=[os.path.join(self.path_to_data, f) for f in file_names]
        
        for i, ii in zip(list_of_paths, file_names):
            import_df=pd.read_csv(i)
            if first_col_frames:
                #import_df.set_index(import_df.columns[0], inplace=True)
                import_df.drop(columns=import_df.columns[0], inplace=True)
            
            if frames_to_seconds:
                import_df.index=import_df.index/fps_rate
                self.x_label='Time (s)'
            else:
                self.x_label='Frames'

            if background_sub:
                background=import_df.iloc[:,background_col]#.to_list()
                import_df=import_df.sub(background, axis=0)
                import_df.drop(columns=import_df.columns[background_col], inplace=True)

            
            self.trial_data_dict[ii]=import_df
        
    def dff_calc(self, method='percentile', percentile=0.2, first_frame=None, last_frame=None):

        if method not in ['median', 'percentile', 'first_frame', 'multi_frame']:
            print('Wrong method defined. Please use one of the following methods: median, percentile, first_frame, multi_frame')
            return

        for i in self.trial_data_dict:
            df=self.trial_data_dict[i]
            if method=='median':
                f0=df.median()
            elif method=='percentile':
                f0=df.quantile(percentile)
            elif method=='first_frame':
                f0=df.iloc[0,:]
            elif method=='multi_frame':
               f0=df.iloc[first_frame:last_frame, :]

            df=(df-f0)/f0
            self.trial_data_dict[i]=df


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

