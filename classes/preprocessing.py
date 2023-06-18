import pandas as pd
import os
import numpy as np
import pickle
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import CheckboxGroup, CustomJS, ColumnDataSource
from bokeh.layouts import column
from bokeh.palettes import Category20  # import a color palette
from matplotlib import pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import zscore
from sklearn import preprocessing

class CalciMetrics:
    def __init__(self, path_to_data):
        self.trial_data_dict={}
        self.path_to_data=path_to_data
        self.dff=False

    def import_data(self, first_col_frames=True, frames_to_seconds=True, fps_rate=1, background_sub=True, background_col=-1):
        file_names=[f for f in os.listdir(self.path_to_data) if f.endswith('.csv')]
        list_of_paths=[os.path.join(self.path_to_data, f) for f in file_names]
        self.frames_to_seconds=frames_to_seconds
        self.fps_rate=fps_rate
        if self.frames_to_seconds:
            x_label='Time (s)'
        else:
            x_label='Frames'
        self.y_label='Fluorescence intensity'
        
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
        print('Data has been successfully imported. The names of your experiments are:')
        for keys, values in self.trial_data_dict.items():
            print(keys) 
        
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
               f0=df.iloc[first_frame:last_frame, :].mean()

            df=(df-f0)/f0
            self.trial_data_dict[i]=df
            self.dff=True
            self.y_label='\u0394 F/F'


    def exclude_rois(self, excluded_rois_dict):

        for i in excluded_rois_dict:
            if set(excluded_rois_dict[i]).issubset(set(self.trial_data_dict[i].columns)):

                cleaned_df=self.trial_data_dict[i].copy()
                exclude_columns=cleaned_df[excluded_rois_dict[i]].columns
                cleaned_df=cleaned_df.drop(columns=exclude_columns)    
                self.trial_data_dict[i]=cleaned_df
            else:
                print('ROIs were already excluded!')
          
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



    def interactive_plot(self, df, dff=False):
        # Create a new figure
        p = figure(width=1000, height=600)
        if self.frames_to_seconds:
            p.xaxis.axis_label = 'Time (s)' 
        else:
            p.xaxis.axis_label = 'Time (s)'
        if self.dff:
            p.yaxis.axis_label = 'delta F/F'
        else:
            p.yaxis.axis_label = 'Fluorescence Intensity'

        # Create a ColumnDataSource from the dataframe
        source = ColumnDataSource(df)

        # Create a color iterator from the palette
        colors = iter(Category20[20])

        # Add a line for each column in the dataframe
        for i, col in enumerate(df.columns):
            p.line(x='index', y=col, line_width=2, alpha=0.7, legend_label=col, source=source, color=next(colors))

        # Create a CheckboxGroup with all columns
        # Set the first checkbox as active, all others inactive
        checkbox_group = CheckboxGroup(labels=list(df.columns), active=[0])

        # Create a callback to hide/show lines
        callback = CustomJS(args=dict(lines=p.renderers, checkbox_group=checkbox_group), code="""
            for (let i = 0; i < lines.length; i++) {
                lines[i].visible = checkbox_group.active.includes(i);
            }
        """)

        # Attach the callback to the CheckboxGroup
        checkbox_group.js_on_change('active', callback)

        # Show the plot and the CheckboxGroup
        show(column(p, checkbox_group))



    def show_plot(self, exp_name):
        output_notebook()
        df=self.trial_data_dict[exp_name]
        self.interactive_plot(df)



    def summary_timecourse(self, exp_name, modus='mean', variance_method='std', start_stim=None, end_stim=None, save_fig=False, save_format='tiff', width=10, hight=5):
        if modus=='mean':
            summary_line=self.summary_dict[exp_name]['mean']
        elif modus=='median':
            summary_line=self.summary_dict[exp_name]['median']
        else:
            print('Please choose modus either "mean" or "median".')
            return

        if variance_method=='std':
            variance=self.summary_dict[exp_name]['std']
        elif variance_method=='sem':
            variance=self.summary_dict[exp_name]['sem']
        else:
            print('Please choose varicane_method either "std" or "sem".')
            return
        fig=plt.figure(figsize=(width,hight))
        plt.plot(summary_line, color='black', linewidth=0.75)
        plt.fill_between(summary_line.index, 
                            summary_line-variance, 
                            summary_line+variance, 
                            alpha=0.5)
        
        plt.axvspan(start_stim, end_stim, color='grey', alpha=0.3, lw=0)

        ax = plt.gca()
        ax.axhline(y=0, color='k', ls='dashed')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        title=' '.join([exp_name[:-4], modus,'+/-', variance_method])
        plt.title(title)
        if save_fig:
            save_path='/'.join([self.path_to_data,'results','fig'])
            if os.path.isdir(save_path):
                plt.savefig((f'{save_path}/{exp_name}_{title}.{save_format}'), format = 'save_format', dpi=300) 
            else:
                os.mkdir(save_path)
                plt.savefig((f'{save_path}/{exp_name}_{title}.{save_format}'), format = 'save_format', dpi=300)
        plt.show()


    def heatmap(self, exp_name, start_stim=None, end_stim=None, save_fig=False, save_format='tiff', width=10, height=8):

        
        if self.frames_to_seconds:
            mesh_df= self.trial_data_dict[exp_name].copy()
            mesh_df.reset_index(drop=True, inplace=True)
            new_index=list(mesh_df.index)
            new_index.append(len(list(mesh_df.index)))
            new_index=pd.Series(new_index)/self.fps_rate
            new_columns=pd.Series(list(range(0, len(mesh_df.columns)+1)))

            fig, ax0 = plt.subplots(figsize=(width,height))
            im = ax0.pcolormesh(new_index,new_columns, (mesh_df.transpose()))
            fig.colorbar(im, ax=ax0, label='\u0394 F/F')
            ax0.set_xlabel('Time (s)')
            ax0.set_ylabel('ROI Number') 
            titel=''.join(['Heatmap of ',exp_name[:-4]])
            plt.title(titel) 
            if start_stim: 
                ax0.axvline(x=start_stim, color='k', ls='dashed', alpha=0.75)
            if end_stim:
                ax0.axvline(x=end_stim, color='k', ls='dashed', alpha=0.75)
            if save_fig:
                save_path='/'.join([self.path_to_data,'results','fig'])
                if os.path.isdir(save_path):
                    plt.savefig((f'{save_path}/{exp_name}_heatmap.{save_format}'), format = 'save_format', dpi=300) 
                else:
                    os.mkdir(save_path)
                    plt.savefig((f'{save_path}/{exp_name}_heatmap.{save_format}'), format = 'save_format', dpi=300)
            plt.show()


        else:
            fig, ax0 = plt.subplots(figsize=(width,height))
            im = ax0.pcolormesh((self.trial_data_dict[exp_name].transpose()))
            fig.colorbar(im, ax=ax0, label='\u0394 F/F')
            ax0.set_xlabel(self.x_label)
            ax0.set_ylabel('ROI Number')
            titel=''.join(['Heatmap of ',exp_name[:-4]])
            plt.title(titel) 

            if start_stim: 
                ax0.axvline(x=start_stim, color='k', ls='dashed', alpha=0.75)
            if end_stim:
                ax0.axvline(x=end_stim, color='k', ls='dashed', alpha=0.75)
            if save_fig:
                save_path='/'.join([self.path_to_data,'results','fig'])
                if os.path.isdir(save_path):
                    plt.savefig((f'{save_path}/{exp_name}_heatmap.{save_format}'), format = 'save_format', dpi=300) 
                else:
                    os.mkdir(save_path)
                    plt.savefig((f'{save_path}/{exp_name}_heatmap.{save_format}'), format = 'save_format', dpi=300) 
            plt.show()



    #def combine_trials(self): TO DO!

    def gaussian_smoothing_viz(self, exp_name, example_col, window_size=10, kernel_std=5):

        cleaned_df=self.trial_data_dict[exp_name]
        column_name=cleaned_df.columns[example_col]
        series_smoothed=cleaned_df[column_name].rolling(window=window_size, win_type='gaussian', center=True).mean(std=kernel_std)
        series_not_smoothed=cleaned_df[column_name]
        window=signal.gaussian(M=window_size, std=kernel_std)

        grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)

        plt.subplot(grid[0, 0])
        plt.plot(series_not_smoothed)
        ax_1=plt.gca()
        ax_1.set_title('Raw Data')
        ax_1.set_xlabel('Time(sec)')
        ax_1.set_ylabel('\u0394F/F')

        plt.subplot(grid[0, 1])
        plt.plot(series_smoothed, color='red') 
        ax_2=plt.gca()
        ax_2.set_title('Smoothed Data')
        ax_2.set_xlabel('Time(sec)')
        ax_2.set_ylabel('\u0394F/F')

        plt.subplot(grid[1, :2])
        plt.plot(window)
        ax_3=plt.gca()
        ax_3.set_title('Gaussian Kernel')
        ax_3.set_xlabel('Sample')
        ax_3.set_ylabel('Weight')
        ax_3.text(0.9,0.9,'window size='+str(window_size)+'\nKernel std='+str(kernel_std),
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax_3.transAxes)

        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.show()


    def smoothing_detrend_viz(self,exp_name,example_col, window_size=20, kernel_std=5, fit_method='exp_decay', detrending_on_baseline=True, base_line_end_time=45):

        column_name=self.trial_data_dict[exp_name].columns[example_col]
        raw_series=self.trial_data_dict[exp_name][column_name]
        series_smoothed=self.trial_data_dict[exp_name][column_name].rolling(window=window_size, win_type='gaussian', center=True).mean(std=kernel_std)
        window=signal.gaussian(M=window_size, std=kernel_std)

        if detrending_on_baseline:
                avg_fluorescence = self.trial_data_dict[exp_name][column_name][:base_line_end_time]      
        else:
                avg_fluorescence = self.trial_data_dict[exp_name][column_name]

        if fit_method=='exp_decay':
            # Define the exponential decay function
            def decay_func(x, a, b, c):
                return a * np.exp(-b * x) + c

            # Fit the exponential decay function to the average fluorescence signal
            x = np.arange(len(avg_fluorescence))
            popt, pcov = curve_fit(decay_func, x, avg_fluorescence)

            # Use the fitted decay function to detrend the fluorescence signal for each ROI
            df_detrend = self.trial_data_dict[exp_name].apply(lambda x: x - decay_func(np.arange(len(x)), *popt), axis=0)
            series_detrend=df_detrend[column_name]
            series_detrend=series_detrend.rolling(window=window_size, win_type='gaussian', center=True).mean(std=kernel_std)

        elif fit_method=='linear':
            
            #input_array=self.trial_data_dict[exp_name][column_name].to_numpy()
            input_array=avg_fluorescence.to_numpy()
            series_detrend_1=signal.detrend(input_array)
            series_detrend=pd.Series(series_detrend_1)
            series_detrend=series_detrend.rolling(window=window_size, win_type='gaussian', center=True).mean(std=kernel_std)

            x = np.arange(len(raw_series))
            #coef = np.polyfit(x, raw_series, deg=1)
            coef = np.polyfit(x, avg_fluorescence, deg=1)
            fit = np.polyval(coef, x)

        else:
            print('Please use the following detrening methods: "linear" or "exp_decay"')
            return

        grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

        plt.subplot(grid[0, 0])
        plt.plot(raw_series)
        ax_1=plt.gca()
        ax_1.set_title('Raw Data')
        ax_1.set_xlabel('Time(sec)')
        ax_1.set_ylabel('\u0394F/F')

        plt.subplot(grid[0, 1])
        plt.plot(series_smoothed, color='red') 
        ax_2=plt.gca()
        ax_2.set_title('Smoothed Data')
        ax_2.set_xlabel('Time(sec)')
        ax_2.set_ylabel('\u0394F/F')

        plt.subplot(grid[1, 0])
        plt.plot(window)
        ax_3=plt.gca()
        ax_3.set_title('Gaussian Kernel')
        ax_3.set_xlabel('Sample')
        ax_3.set_ylabel('Weight')
        ax_3.text(0.5,0.1,'window size='+str(window_size)+'\nKernel std='+str(kernel_std),
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax_3.transAxes)

        if fit_method=='exp_decay':
            plt.subplot(grid[0, 2])
            plt.plot(series_detrend, color='red') 
            ax_4=plt.gca()
            ax_4.set_title('exponential decay detrendet and Smoothed Data')
            ax_4.set_xlabel('Time(sec)')
            ax_4.set_ylabel('\u0394F/F')

            if detrending_on_baseline:
                avg_fluorescence = self.trial_data_dict[exp_name][column_name]
                x = np.arange(len(avg_fluorescence))
                
            avg=pd.Series(avg_fluorescence)
            decay_fit=pd.Series(decay_func(x, *popt))
            decay_fit.index=decay_fit.index/self.fps_rate

            plt.subplot(grid[1, 1:])
            plt.plot(avg, label='Mean Fluorescence')
            plt.plot(decay_fit, label='Fitted Function')
            ax_5=plt.gca()
            ax_5.set_xlabel('Time (frames)')
            ax_5.set_ylabel('Fluorescence Intensity')
            ax_5.legend()
            ax_5.set_title('Exponential decay fit vs. real data')

        elif fit_method=='linear':
            series_detrend.index=series_detrend.index/self.fps_rate
            plt.subplot(grid[0, 2])
            plt.plot(series_detrend, color='red') 
            ax_4=plt.gca()
            ax_4.set_title('linear detrendet and Smoothed Data')
            ax_4.set_xlabel('Time(sec)')
            ax_4.set_ylabel('\u0394F/F')

            fit=pd.Series(fit)
            fit.index=fit.index/self.fps_rate
            plt.subplot(grid[1, 1:])
            plt.plot(raw_series, label='Mean Fluorescence')
            plt.plot(fit, label='Fitted Function')
            ax_5=plt.gca()
            ax_5.set_xlabel('Time (frames)')
            ax_5.set_ylabel('Fluorescence Intensity')
            ax_5.legend()
            ax_5.set_title('linear fit vs. real data')
   
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.show()


        