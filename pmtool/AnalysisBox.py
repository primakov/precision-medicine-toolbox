import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly
import plotly.subplots
import matplotlib.pyplot as plt
from statsmodels.stats import multitest
import os
import scipy as sp
import math
import scipy.stats
from operator import le, gt
from sklearn.metrics import auc
import plotly.io as pio
from pmtool.FeaturesSet import FeaturesSet

class AnalysisBox(FeaturesSet):

    '''This module is inherited from FeaturesSet class and allows for preliminary statistical analysis of the numerical features.'''

    def handle_nan(self, axis: int=1, how: str='any', mode: str='delete'):
        """Handle the missing values.

        Arguments:
            axis: Determines if patients (0) or variables (1) with the missing values have to be fixed.
            how: Determines if handling is needed when there is at least one missing value ('any') or all of them are missing ('all').
            mode: Determines the strategy: 'delete' will delete the variable/patient, 'fill' will fill a missing value with the imputation method.
        """
        if mode == 'delete':
            self._feature_dataframe.dropna(axis=axis, how=how, inplace=True)
            self._feature_outcome_dataframe.dropna(axis=axis, how=how, inplace=True)
            self._feature_column = list(self._feature_dataframe.columns)
            self._patient_name = list(self._feature_outcome_dataframe.index)
            if self._outcome_column in self._feature_column:
                self._feature_column.remove(self._outcome_column)
            self._outcome = self._feature_outcome_dataframe[self._outcome_column]
            self._class_label = pd.unique(np.array(list(self._outcome)))
            self._class_label.sort()
            data_balance = []
            for label_name in self._class_label:
                data_balance.append(np.sum(np.array(list(self._outcome)) == label_name)/len(self._outcome))
            print('Number of observations: {}\nClass labels: {}\nClasses balance: {}'.format(len(self._outcome),
                                                                                        self._class_label,
                                                                                        data_balance))
        if mode == 'fill':
            print('Not implemented yet')

        return None

    def handle_constant(self):
        """Drop the features with the constant values."""

        constant_features = self._feature_dataframe.columns[self._feature_dataframe.nunique() <= 1]
        self._feature_dataframe.drop(constant_features, axis=1, inplace=True)
        self._feature_outcome_dataframe.drop(constant_features, axis=1, inplace=True)
        self._feature_column = list(self._feature_dataframe.columns)
        if self._outcome_column in self._feature_column:
            self._feature_column.remove(self._outcome_column)
        self._outcome = self._feature_outcome_dataframe[self._outcome_column]
        self._class_label = pd.unique(np.array(list(self._outcome)))
        self._class_label.sort()
        data_balance = []
        for label_name in self._class_label:
            data_balance.append(np.sum(np.array(list(self._outcome)) == label_name) / len(self._outcome))
        print('Number of observations: {}\nClass labels: {}\nClasses balance: {}'.format(len(self._outcome),
                                                                                         self._class_label,
                                                                                         data_balance))

        return None

    def plot_distribution(self, features_to_plot: list=[], binary_classes_to_plot: list=[]):
        """Plot distribution of the feature values in classes into interactive .html report.

        Arguments:
            features_to_plot: List of specific features to be selected (otherwise selects all the numerical features).
            binary_classes_to_plot: List, containing 2 classes of interest, if the dataset is multi-class.
        """

        if len(self._outcome) > 0:
            if len(binary_classes_to_plot) == 2:
                if (binary_classes_to_plot[0] in self._class_label) & (binary_classes_to_plot[1] in self._class_label):
                    if not features_to_plot:
                        features_to_plot = self._feature_column
                    num_features = []
                    for feature in features_to_plot:
                        if self._feature_dataframe[feature].dtype != 'object':
                            num_features.append(feature)
                    cols = 4
                    rows = len(num_features) // 4 + 1
                    num_features_tuple = tuple(num_features)
                    fig = plotly.subplots.make_subplots(rows=rows, cols=cols, subplot_titles=num_features_tuple)
                    counter = 0
                    for feature in num_features:
                        c = counter % 4 + 1
                        r = counter // 4 + 1
                        fig.append_trace(go.Histogram(
                            x=self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                                  binary_classes_to_plot[0]][feature],
                            opacity=0.75,
                            name=str(binary_classes_to_plot[0]),
                            marker={'color': 'magenta'},
                            showlegend=(counter == 0)),
                            r, c)
                        fig.append_trace(go.Histogram(
                            x=self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                                  binary_classes_to_plot[1]][feature],
                            opacity=0.75,
                            name=str(binary_classes_to_plot[1]),
                            marker={'color': 'orange'},
                            showlegend=(counter == 0)),
                            r, c)
                        fig.update_xaxes(title_text='values', title_font={"size": 10}, title_standoff=5, row=r, col=c,
                                         showgrid=False, zeroline=False)
                        fig.update_yaxes(title_text='count', title_font={"size": 10}, title_standoff=0, row=r, col=c,
                                         showgrid=False, zeroline=False)
                        fig.layout.update(go.Layout(barmode='overlay'))
                        counter += 1
                    for i in fig['layout']['annotations']:
                        i['font'] = dict(size=10)
                    fig.update_layout(title_text='Features binary distribution in classes',
                                      height=rows * 250, width=1250)
                    plotly.offline.plot(fig,
                                        filename=os.path.splitext(self._feature_path)[0] + '_distr.html',
                                        config={'scrollZoom': True})
                else:
                    print('Wrong class label(s).')
            else:

                color_scheme = ['magenta', 'orange', 'cyan', 'yellow', 'lime',
                                'blue', 'red', 'green', 'darkviolet', 'saddlebrown']
                if not features_to_plot:
                    features_to_plot = self._feature_column
                num_features = []
                for feature in features_to_plot:
                    if self._feature_dataframe[feature].dtype != 'object':
                        num_features.append(feature)
                cols = 4
                rows = len(num_features) // 4 + 1
                num_features_tuple = tuple(num_features)
                fig = plotly.subplots.make_subplots(rows=rows, cols=cols, subplot_titles=num_features_tuple)
                counter = 0
                for feature in num_features:
                    c = counter % 4 + 1
                    r = counter // 4 + 1
                    counter_colors = 0
                    for cl in self._class_label:
                        fig.append_trace(go.Histogram(
                            x=self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                                  cl][feature],
                            opacity=0.5,
                            name=str(cl),
                            marker={'color': color_scheme[counter_colors]},
                            showlegend=(counter == 0)),
                            r, c)
                        counter_colors += 1

                    fig.update_xaxes(title_text='values', title_font={"size": 10}, title_standoff=5, row=r, col=c,
                                     showgrid=False, zeroline=False)
                    fig.update_yaxes(title_text='count', title_font={"size": 10}, title_standoff=0, row=r, col=c,
                                     showgrid=False, zeroline=False)
                    fig.layout.update(go.Layout(barmode='overlay'))
                    counter += 1
                for i in fig['layout']['annotations']:
                    i['font'] = dict(size=10)
                fig.update_layout(title_text='Features binary distribution in classes',
                                  height=rows * 250, width=1250)
                plotly.offline.plot(fig,
                                    filename=os.path.splitext(self._feature_path)[0] + '_distr.html',
                                    config={'scrollZoom': True})

        else:
            print('Outcome column should be presented')

        return None

    def plot_correlation_matrix(self, features_to_plot: list=[]):
        """Plot correlation (Spearman's) matrix for the features into interactive .html report.

        Arguments:
            features_to_plot: List of specific features to be selected (otherwise selects all the numerical features).
        """

        pio.renderers.default = 'iframe'

        if not features_to_plot:
            features_to_plot = self._feature_column

        num_features = []
        for feature in features_to_plot:
            if self._feature_dataframe[feature].dtype != 'object':
                num_features.append(feature)

        data = go.Heatmap(
            z=np.abs(np.array(self._feature_dataframe[num_features].corr(method='spearman'))),
            x=num_features,
            y=num_features,
            colorbar=dict(title='Spearman corr'),
            hovertemplate='feature_1: %{x}<br>feature_2: %{y}<br>r_Spearman: %{z}<extra></extra>'
        )

        layout = {"title": "Features correlation matrix"}
        fig = go.Figure(data=data, layout=layout)
        fig.update_xaxes(tickfont=dict(size=7))
        fig.update_yaxes(tickfont=dict(size=7))
        fig.update_layout(height=len(num_features) * 20 + 250, width=len(num_features) * 20 + 250)
        plotly.offline.plot(fig,
                            filename=os.path.splitext(self._feature_path)[0] + '_corr.html',
                            config={'scrollZoom': True})

        return None

    def __get_color(self, v, th):
        if abs(v) <= th:
            return 'orange'
        else:
            return 'purple'

    def __get_MW_p(self, ftrs, binary_classes_to_plot, p_threshold=0.05):
        p_MW = []
        df_0 = self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                   binary_classes_to_plot[0]]
        df_1 = self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                   binary_classes_to_plot[1]]
        for feature in ftrs:
            s, p = sp.stats.mannwhitneyu(df_0[feature].astype(float),
                                         df_1[feature].astype(float),
                                         alternative='two-sided')
            p_MW.append(p)
        reject, p_MW_corr, a_S_corr, a_B_corr = multitest.multipletests(p_MW,
                                                                        alpha=p_threshold,
                                                                        method='bonferroni')
        return p_MW_corr

    def plot_MW_p(self, features_to_plot: list=[], binary_classes_to_plot: list=[], p_threshold: float=0.05):
        """Plot two-sided Mann-Whitney U test p-values for comparison of features values means in 2 classes (with correction for multiple testing) into interactive .html report.

        Arguments:
            features_to_plot: List of specific features to be selected (otherwise selects all the numerical features).
            binary_classes_to_plot: List, containing 2 classes of interest, if the dataset is multi-class.
            p_threshold: Significance level.
        """

        if len(self._outcome) > 0:
            if len(self._class_label) == 2:
                binary_classes_to_plot = self._class_label
            else:
                if len(binary_classes_to_plot) != 2:
                    print('Only binary class labels are supported.')
                    return
                elif not ((binary_classes_to_plot[0] in self._class_label) & (binary_classes_to_plot[1] in self._class_label)):
                    print('Wrong class label(s).')
                    return

            if not features_to_plot:
                features_to_plot = self._feature_column

            num_features = []
            for feature in features_to_plot:
                if self._feature_dataframe[feature].dtype != 'object':
                    num_features.append(feature)
            p_MW_corr = self.__get_MW_p(ftrs=num_features,
                                        binary_classes_to_plot=binary_classes_to_plot,
                                        p_threshold=p_threshold)

            colors = [self.__get_color(v, p_threshold) for v in p_MW_corr]
            shapes = [{'type': 'line',
                        'xref': 'x',
                        'yref': 'y',
                        'x0': p_threshold,
                        'y0': 0,
                        'x1': p_threshold,
                        'y1': len(num_features)}]
            annotations = [
                go.layout.Annotation(
                    x=math.log10(p_threshold),
                    y=len(num_features),
                    text="alpha="+str(p_threshold),
                    showarrow=True, arrowhead=7, ax=100, ay=0
                )
            ]
            layout = plotly.graph_objs.Layout(shapes=shapes)

            fig = go.Figure(
                [go.Bar(x=list(p_MW_corr), y=num_features, marker={'color': colors}, orientation='h')],
                layout=layout)
            fig.update_yaxes(tickfont=dict(size=7))
            fig.update_xaxes(tickfont=dict(size=7))
            fig.update_layout(title_text='The p-values for Mann-Whitney test (Bonferroni corrected)',
                                height=len(num_features) * 20+250, width=750,
                                xaxis_type="log", annotations=annotations,
                                xaxis={"mirror": "allticks", 'side': 'top', 'dtick': 1, 'showgrid': True}
                                )
            plotly.offline.plot(fig,
                                filename=os.path.splitext(self._feature_path)[0] + '_MW.html',
                                config={'scrollZoom': True})
        else:
            print('Outcome column should be presented')
        return None

    def __get_univar_fprs_tprs(self, ftr, binary_classes_to_plot):

        mean_0 = self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                     binary_classes_to_plot[0]][ftr].mean()
        mean_1 = self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                     binary_classes_to_plot[1]][ftr].mean()
        feature_min = np.min(np.array(self._feature_outcome_dataframe[ftr]))
        feature_max = np.max(np.array(self._feature_outcome_dataframe[ftr]))
        step = (feature_max - feature_min) / 100
        x = np.array(self._feature_outcome_dataframe[ftr])
        n_0 = len(self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                      binary_classes_to_plot[0]])
        n_1 = len(self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                      binary_classes_to_plot[1]])
        th = []
        tprs = []
        fprs = []
        op = gt
        if mean_0 > mean_1:
            op = le

        for i in range(0, 102):
            t = feature_min + i * step
            th.append(t)
            y_pred = op(x, t).astype(int)
            tp = np.sum(y_pred[np.where(self._feature_outcome_dataframe[self._outcome_column] == binary_classes_to_plot[1])])
            fp = np.sum(y_pred[np.where(self._feature_outcome_dataframe[self._outcome_column] == binary_classes_to_plot[0])])
            tpr = tp / n_1
            fpr = fp / n_0
            tprs.append(tpr)
            fprs.append(fpr)

        return fprs, tprs

    def plot_univariate_roc(self, features_to_plot: list=[], binary_classes_to_plot: list=[], auc_threshold: float=0.75):
        """Plot univariate ROC curves (with AUC calculation) for threshold binary classifier, based of each feature separately into interactive .html report.

        Arguments:
            features_to_plot: List of specific features to be selected (otherwise selects all the numerical features).
            binary_classes_to_plot: List, containing 2 classes of interest in case of multi-class data.
            auc_threshold: Threshold value for ROC AUC to be highlighted.
        """

        if len(self._outcome) > 0:
            if len(self._class_label) == 2:
                binary_classes_to_plot = self._class_label
            else:
                if len(binary_classes_to_plot) != 2:
                    print('Only binary class labels are supported.')
                    return
                elif not ((binary_classes_to_plot[0] in self._class_label) & (
                        binary_classes_to_plot[1] in self._class_label)):
                    print('Wrong class label(s).')
                    return

            if not features_to_plot:
                features_to_plot = self._feature_column

            num_features = []
            for feature in features_to_plot:
                if self._feature_outcome_dataframe[feature].dtype != 'object':
                    num_features.append(feature)

            cols = 4
            rows = len(num_features) // 4 + 1
            num_features_tuple = tuple(num_features)

            fig = plotly.subplots.make_subplots(rows=rows, cols=cols, subplot_titles=num_features_tuple)
            counter = 0

            for feature in num_features:
                c = counter % 4 + 1
                r = counter // 4 + 1
                fprs, tprs = self.__get_univar_fprs_tprs(ftr=feature, binary_classes_to_plot=binary_classes_to_plot)
                univar_auc = auc(fprs, tprs)
                fig.append_trace(go.Scatter(x=fprs, y=tprs,
                                            name='ROC',
                                            marker={'color': self.__get_color(univar_auc, auc_threshold)}),
                                    r, c)
                fig.append_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                            name='Chance', marker={'color': 'grey'},
                                            mode='lines+markers+text',
                                            text=[' ROC AUC=%0.2f' % univar_auc, ''], textposition='middle right'),
                                    r, c)

                fig.update_xaxes(title_text='FPR', title_font={"size": 10}, title_standoff=5, row=r, col=c,
                                 showgrid=False, zeroline=False)
                fig.update_yaxes(title_text='TPR', title_font={"size": 10}, title_standoff=0, row=r, col=c,
                                 showgrid=False, zeroline=False)
                counter += 1

            for i in fig['layout']['annotations']:
                i['font'] = dict(size=10)

            fig.update_layout(title_text='Features univariate ROC-curves:' + str(binary_classes_to_plot),
                              height=rows * 250, width=1250, showlegend=False)
            plotly.offline.plot(fig,
                                filename=os.path.splitext(self._feature_path)[0] + '_roc-univar.html',
                                config={'scrollZoom': True})
        else:
            print('Outcome column should be presented')

        return None

    def calculate_basic_stats(self, volume_feature: str=''):
        """Calculate basic statistical scores (such as: number of missing values, mean, std, min, max, Mann-Whitney test p-values for binary classes, univariate ROC AUC for binary classes, Spearman's correlation with volume if volumetric feature name is sent to function) for each feature and save it to .csv file.

        Arguments:
            volume_feature: Name of the feature, which is considered as volume.
        """

        num_features = []
        for feature in self._feature_column:
            if self._feature_dataframe[feature].dtype != 'object':
                num_features.append(feature)

        frame = {'NaN': self._feature_dataframe[num_features].isnull().sum(axis=0),
                 'Mean': self._feature_dataframe[num_features].mean(axis=0),
                 'Std': self._feature_dataframe[num_features].std(axis=0),
                 'Min': self._feature_dataframe[num_features].min(axis=0),
                 'Max': self._feature_dataframe[num_features].max(axis=0)}
        stats_dataframe = pd.DataFrame(frame)

        if len(self._outcome) > 0:
            if len(self._class_label) == 2:
                p_MW_corr = self.__get_MW_p(ftrs=num_features, binary_classes_to_plot=self._class_label)
                univar_auc = []
                for feature in num_features:
                    fprs, tprs = self.__get_univar_fprs_tprs(ftr=feature, binary_classes_to_plot=self._class_label)
                    univar_auc.append(auc(fprs, tprs))
                stats_dataframe_ext = pd.DataFrame({'p_MW_corrected': p_MW_corr,
                                                    'univar_auc': univar_auc},
                                                   index=num_features)

                stats_dataframe = pd.concat([stats_dataframe, stats_dataframe_ext], axis=1)

        if volume_feature:
            vol_corr = []
            for feature in num_features:
                vol_corr.append(sp.stats.spearmanr(self._feature_dataframe[feature],
                                                   self._feature_dataframe[volume_feature])[0])
            stats_dataframe = pd.concat([stats_dataframe,
                                         pd.DataFrame({'volume_corr': vol_corr}, index=num_features)],
                                        axis=1)

        stats_dataframe.to_excel(os.path.splitext(self._feature_path)[0] + '_basic_stats.xlsx')

        return None

    def __get_univar_prec_rec(self, ftr):

        mean_0 = self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                     self._class_label[0]][ftr].mean()
        mean_1 = self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                     self._class_label[1]][ftr].mean()
        feature_min = np.min(np.array(self._feature_outcome_dataframe[ftr]))
        feature_max = np.max(np.array(self._feature_outcome_dataframe[ftr]))
        step = (feature_max - feature_min) / 100
        x = np.array(self._feature_outcome_dataframe[ftr])
        n_1 = len(self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                      self._class_label[1]])
        th = []
        precs = []
        recs = []
        op = gt
        if mean_0 > mean_1:
            op = le

        for i in range(0, 102):
            t = feature_min + i * step
            th.append(t)
            y_pred = op(x, t).astype(int)
            if np.sum(y_pred>0):
                tp = np.sum(y_pred[np.where(self._feature_outcome_dataframe[self._outcome_column] == self._class_label[1])])
                prec = tp / np.sum(y_pred)
                rec = tp / n_1
                precs.append(prec)
                recs.append(rec)

        return precs, recs

    def volume_analysis(self, volume_feature: str='', auc_threshold: float=0.75, features_to_plot: list=[], corr_threshold: float=0.75):
        '''Calculate features correlation (Spearman’s) with volume and plot volume-based precision-recall curve.

        Arguments:
            volume_feature: Name of the feature, which is considered as volume.
            auc_threshold: Threshold value for area under precision-recall curve to be highlighted.
            features_to_plot: Specific features to be selected (otherwise selects all the numerical features)
            corr_threshold: Threshold value for absolute value for Spearman’s correlation coefficient to be considered as ‘strong correlation’.
        '''
        if volume_feature:
            if volume_feature in self._feature_column:
                if len(self._outcome) > 0:
                    if len(self._class_label) == 2:

                        precs, recs = self.__get_univar_prec_rec(ftr=volume_feature)
                        univar_auc = auc(recs, precs)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=recs, y=precs,
                                                 name='PRC',
                                                 marker={'color': self.__get_color(univar_auc, auc_threshold)}))
                        fig.add_trace(go.Scatter(x=[0], y=[0],
                                                 name='AUC', marker={'color': 'grey'},
                                                 mode='text',
                                                 text=['AUC=%0.2f' % univar_auc, ''], textposition='top right'))
                        fig.update_xaxes(title_text='recall', title_font={"size": 10}, title_standoff=5,
                                         showgrid=True, zeroline=False, range=[0, 1])
                        fig.update_yaxes(title_text='precision', title_font={"size": 10}, title_standoff=5,
                                         showgrid=True, zeroline=False, range=[0, 1])

                        fig.update_layout(title_text='Volume precision-recall curve', showlegend=False, width=500,
                                          height=500)
                        plotly.offline.plot(fig,
                                            filename=os.path.splitext(self._feature_path)[0] + '_volume_PRC.html',
                                            config={'scrollZoom': True})

                if not features_to_plot:
                    features_to_plot = self._feature_column
                num_features = []
                corr_vol = []
                for feature in features_to_plot:
                    if self._feature_outcome_dataframe[feature].dtype != 'object':
                        num_features.append(feature)
                        corr_vol.append(sp.stats.spearmanr(self._feature_dataframe[feature],
                                                           self._feature_dataframe[volume_feature])[0])
                colors = [self.__get_color(v, corr_threshold) for v in corr_vol]
                shapes = [{'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': corr_threshold,
                           'y0': 0,
                           'x1': corr_threshold,
                           'y1': len(num_features)}]
                annotations = [
                    go.layout.Annotation(
                        x=corr_threshold,
                        y=len(num_features),
                        text="r_S=" + str(corr_threshold),
                        showarrow=True, arrowhead=7, ax=100, ay=0
                    )
                ]
                layout = plotly.graph_objs.Layout(shapes=shapes)
                fig = go.Figure(layout=layout)
                fig.add_trace(go.Bar(x=list(np.abs(corr_vol)), y=num_features, marker={'color': colors},
                                     orientation='h', name='Spearman correlation'))
                fig.update_yaxes(tickfont=dict(size=7))
                fig.update_xaxes(tickfont=dict(size=7))
                fig.update_layout(annotations=annotations,
                                  xaxis={"mirror": "allticks", 'side': 'top', 'dtick': 1, 'showgrid': True},
                                  title_text='Volume Spearman correlation', showlegend=False, width=750,
                                  height=len(num_features) * 20+250)
                plotly.offline.plot(fig, filename=os.path.splitext(self._feature_path)[0] + '_volume_corr.html',
                                    config={'scrollZoom': True})

        return None

    def split_feature_types(self):
        """Split all features into lists by feature type."""
        
        shape_features = []
        first_order_features = []
        glcm_features = []
        glrlm_features = []
        glszm_features = []
        gldm_features = []
        log_features = []
        wavelet_features = []
         
        for feature in self._feature_column:
            if "original_shape" in feature:
                shape_features.append(feature)
            if "original_firstorder" in feature:
                first_order_features.append(feature)
            if "original_glcm" in feature:
                glcm_features.append(feature)
            if "original_glrlm" in feature:
                glrlm_features.append(feature)
            if "original_glszm" in feature:
                glszm_features.append(feature)
            if "original_gldm" in feature:
                gldm_features.append(feature)
            if "log-sigma" in feature:
                log_features.append(feature)
            if "wavelet" in feature:
                wavelet_features.append(feature)
                
        return shape_features, first_order_features, glcm_features, glrlm_features, glszm_features, gldm_features, log_features, wavelet_features


    def calculate_pearson_correlation(self, output_path = False, binary_classes_to_plot: list=[]):
        """Calculate the Pearson correlation coefficient with the possibility to export this to an Excel worksheet.

        Arguments:
            output_path: Can be added to export the results to an Excel worksheet.
            binary_classes_to_plot: List, containing 2 classes of interest, if the dataset is multi-class.
        """
        
        pearson_correlation = []
        
        if len(self._outcome) > 0:
            if len(self._class_label) == 2:
                binary_classes_to_plot = self._class_label
            else:
                if len(binary_classes_to_plot) != 2:
                    print('Only binary class labels are supported.')
                    return
                elif not ((binary_classes_to_plot[0] in self._class_label) & (
                        binary_classes_to_plot[1] in self._class_label)):
                    print('Wrong class label(s).')
                    return
        
        shape_features, first_order_features, glcm_features, glrlm_features, glszm_features, gldm_features, log_features, wavelet_features = self.split_feature_types()
        features_to_use = shape_features + first_order_features + glcm_features + glrlm_features + glszm_features + gldm_features + log_features + wavelet_features
        
        for feature in features_to_use:
                value_list0 = (self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                             binary_classes_to_plot[0]]).loc[:, feature]
                value_list1 = (self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                             binary_classes_to_plot[1]]).loc[:, feature]
                if len(value_list0) != len(value_list1) :
                    print("This correlation is only available for binary classes with the same number of samples.")
                    return
                pearson_correlation.append(scipy.stats.pearsonr(value_list0, value_list1))
        pearson_correlation=np.array(pearson_correlation)
        
        pearson_r = pearson_correlation[:,0].tolist()
        pearson_df = pd.DataFrame({"feature":features_to_use, "Pearson r": pearson_r})
        
        if isinstance(output_path, str):
            if "\\" in output_path: 
                pearson_df.to_excel(output_path + "\\"+ 'Pearson correlation.xlsx')
            elif "/" in output_path: 
                pearson_df.to_excel(output_path + "/"+ 'Pearson correlation.xlsx')    
        
        
        return pearson_df

    def calculate_spearman_correlation(self, output_path = False, binary_classes_to_plot: list=[]):         
        """Calculate Spearman's Rank correlation coefficient with the possibility to export this to an Excel worksheet.

        Arguments:
            output_path: Can be added to export the results to an Excel worksheet.
            binary_classes_to_plot: List, containing 2 classes of interest, if the dataset is multi-class.
        """
        
        spearman_correlation = []
        
        if len(self._outcome) > 0:
            if len(self._class_label) == 2:
                binary_classes_to_plot = self._class_label
            else:
                if len(binary_classes_to_plot) != 2:
                    print('Only binary class labels are supported.')
                    return
                elif not ((binary_classes_to_plot[0] in self._class_label) & (
                        binary_classes_to_plot[1] in self._class_label)):
                    print('Wrong class label(s).')
                    return
        
        shape_features, first_order_features, glcm_features, glrlm_features, glszm_features, gldm_features, log_features, wavelet_features = self.split_feature_types()
        features_to_use = shape_features + first_order_features + glcm_features + glrlm_features + glszm_features + gldm_features + log_features + wavelet_features
        
        for feature in features_to_use:
                value_list0 = (self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                             binary_classes_to_plot[0]]).loc[:, feature]
                value_list1 = (self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                             binary_classes_to_plot[1]]).loc[:, feature]
                if len(value_list0) != len(value_list1) :
                    print("This correlation is only available for binary classes with the same number of samples.")
                    return
                spearman_correlation.append(scipy.stats.spearmanr(value_list0, value_list1))
        spearman_correlation=np.array(spearman_correlation)
        
        spearman_r = spearman_correlation[:,0].tolist()
        spearman_df = pd.DataFrame({"feature":features_to_use, "Spearman r": spearman_r})
        
        if isinstance(output_path, str):
            if "\\" in output_path: 
                spearman_df.to_excel(output_path + "\\"+ 'Spearman correlation.xlsx')
            elif "/" in output_path: 
                spearman_df.to_excel(output_path + "/"+ 'Spearman correlation.xlsx')    
        
        
        return spearman_df
    
    def calculate_CCC(self, output_path = False, binary_classes_to_plot: list=[]):
        """Calculate Lin's concordance correlation coefficient (CCC) with the possibility to export this to an Excel worksheet.

        Arguments:
            output_path: Can be added to export the results to an Excel worksheet.
            binary_classes_to_plot: List, containing 2 classes of interest, if the dataset is multi-class.
        """
        
        CCC = []
        
        if len(self._outcome) > 0:
            if len(self._class_label) == 2:
                binary_classes_to_plot = self._class_label
            else:
                if len(binary_classes_to_plot) != 2:
                    print('Only binary class labels are supported.')
                    return
                elif not ((binary_classes_to_plot[0] in self._class_label) & (
                        binary_classes_to_plot[1] in self._class_label)):
                    print('Wrong class label(s).')
                    return
                
        shape_features, first_order_features, glcm_features, glrlm_features, glszm_features, gldm_features, log_features, wavelet_features = self.split_feature_types()
        features_to_use = shape_features + first_order_features + glcm_features + glrlm_features + glszm_features + gldm_features + log_features + wavelet_features
        
        for feature in features_to_use:
                value_list0 = (self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                             binary_classes_to_plot[0]]).loc[:, feature]
                value_list1 = (self._feature_outcome_dataframe.loc[self._feature_outcome_dataframe[self._outcome_column] ==
                                                             binary_classes_to_plot[1]]).loc[:, feature]
                if len(value_list0) != len(value_list1) :
                    print("This correlation is only available for binary classes with the same number of samples.")
                    return
                pearson_r = scipy.stats.pearsonr(value_list0, value_list1)
                mean0=np.mean(value_list0)
                mean1=np.mean(value_list1)
                st_dev0=np.std(value_list0)
                st_dev1=np.std(value_list1)
                ccc_feature = (2 * pearson_r[0] * st_dev0 * st_dev1) / (st_dev0**2 + st_dev1**2 + (mean0 - mean1)**2)
                CCC.append(ccc_feature)
                
        CCC_df = pd.DataFrame({"feature":features_to_use, "CCC": CCC})
        
        if isinstance(output_path, str):
            if "\\" in output_path: 
                CCC_df.to_excel(output_path + "\\"+ 'CCC.xlsx')
            elif "/" in output_path: 
                CCC_df.to_excel(output_path + "/"+ 'CCC.xlsx')    
        

        return CCC_df
        
      
    def plot_correlations(self, correlation_type= "pearson", threshold: float=0.9, exclude_feature_groups: list=[], binary_classes_to_plot: list=[]):
        """Plot the correlation of your choice.

        Arguments:
            correlation_type: Choose Pearson, Spearman, or Lin's concordance correlation coefficient (CCC) to plot (Pearson is chosen by default)
            threshold: Correlation threshold level, add a line and change bar colors at this level.
            exclude_feature_groups: List of feature groups not to plot (otherwise all feature groups will be plotted).
            binary_classes_to_plot: List, containing 2 classes of interest, if the dataset is multi-class.
        """
        
        feature_group_list = ["shape_features", "first_order_features", "glcm_features", "glrlm_features", "glszm_features", "gldm_features", "log_features", "wavelet_features"]
        shape_features, first_order_features, glcm_features, glrlm_features, glszm_features, gldm_features, log_features, wavelet_features = self.split_feature_types()
        list_of_groups = [shape_features, first_order_features, glcm_features, glrlm_features, glszm_features, gldm_features, log_features, wavelet_features]
        
        if len(self._outcome) > 0:
            if len(self._class_label) == 2:
                binary_classes_to_plot = self._class_label
            else:
                if len(binary_classes_to_plot) != 2:
                    print('Only binary class labels are supported.')
                    return
                elif not ((binary_classes_to_plot[0] in self._class_label) & (
                        binary_classes_to_plot[1] in self._class_label)):
                    print('Wrong class label(s).')
                    return
        
        if not 0 <= threshold <= 1:
            print ("The threshold needs to be between 0 and 1")
            return
        
        if bool(exclude_feature_groups) == True:
            feature_check = all(features in feature_group_list for features in exclude_feature_groups)
            if feature_check == False:
                print("Please input a list of one or none of the following feature group names to exclude: \n    \"shape_features\", \"first_order_features\", \"glcm_features\", \"glrlm_features\", \"glszm_features\", \"gldm_features\", \"log_features\", \"wavelet_features\"")
                return
        
        if correlation_type.lower() == "pearson":
            correlation_df = self.calculate_pearson_correlation()
        elif correlation_type.lower() == "spearman":
            correlation_df = self.calculate_spearman_correlation()
        elif correlation_type.lower() == "ccc": 
            correlation_df = self.calculate_CCC()
        else:
            print("Please select one of the following feature types: \n    \"pearson\", \"spearman\", \"CCC\"")
            return            
        
        correlation_df.set_index("feature", inplace = True)
        
        if "shape_features" in exclude_feature_groups:
            list_of_groups.remove(shape_features)
        else:
            shape_correlation = []
            for feature in shape_features:
                shape_correlation.append(correlation_df.loc[feature][0])
            shape_clean = [substring.replace('original_shape_', '') for substring in shape_features]
            shape_df= pd.DataFrame({"feature":shape_clean, "correlation": shape_correlation}) 
            shape_df_sorted = shape_df.sort_values(by=['correlation'], ascending=False)
            
            y_shape = np.array(shape_df_sorted['feature'])
            x_shape = np.array(shape_df_sorted['correlation'])
            shape_mask = x_shape > threshold
            shape_mask_off = x_shape < threshold
        
        if "first_order_features" in exclude_feature_groups:
            list_of_groups.remove(first_order_features)
        else:
            first_order_correlation = []
            for feature in first_order_features:
                first_order_correlation.append(correlation_df.loc[feature][0])
            first_order_clean = [substring.replace('original_firstorder_', '') for substring in first_order_features]
            first_order_df= pd.DataFrame({"feature":first_order_clean, "correlation": first_order_correlation})
            first_order_df_sorted = first_order_df.sort_values(by=['correlation'], ascending=False)
            
            y_first_order = np.array(first_order_df_sorted['feature'])
            x_first_order = np.array(first_order_df_sorted['correlation'])
            first_order_mask = x_first_order > threshold
            first_order_mask_off = x_first_order < threshold
        
        if "glcm_features" in exclude_feature_groups:
            list_of_groups.remove(glcm_features)
        else:
            glcm_correlation = []
            for feature in glcm_features:
                glcm_correlation.append(correlation_df.loc[feature][0])
            glcm_clean = [substring.replace('original_glcm_', '') for substring in glcm_features]    
            glcm_df= pd.DataFrame({"feature":glcm_clean, "correlation": glcm_correlation})
            glcm_df_sorted = glcm_df.sort_values(by=['correlation'], ascending=False)
            
            y_glcm = np.array(glcm_df_sorted['feature'])
            x_glcm = np.array(glcm_df_sorted['correlation'])
            glcm_mask = x_glcm > threshold
            glcm_mask_off = x_glcm < threshold
        
        if "glrlm_features" in exclude_feature_groups:
            list_of_groups.remove(glrlm_features)
        else:
            glrlm_correlation = []
            for feature in glrlm_features:
                glrlm_correlation.append(correlation_df.loc[feature][0])
            glrlm_clean = [substring.replace('original_glrlm_', '') for substring in glrlm_features]
            glrlm_df= pd.DataFrame({"feature":glrlm_clean, "correlation": glrlm_correlation})
            glrlm_df_sorted = glrlm_df.sort_values(by=['correlation'], ascending=False)
            
            y_glrlm = np.array(glrlm_df_sorted['feature'])
            x_glrlm = np.array(glrlm_df_sorted['correlation'])
            glrlm_mask = x_glrlm > threshold
            glrlm_mask_off = x_glrlm < threshold
        
        if "glszm_features" in exclude_feature_groups:
            list_of_groups.remove(glszm_features)
        else:
            glszm_correlation = []
            for feature in glszm_features:
                glszm_correlation.append(correlation_df.loc[feature][0])
            glszm_clean = [substring.replace('original_glszm_', '') for substring in glszm_features]
            glszm_df= pd.DataFrame({"feature":glszm_clean, "correlation": glszm_correlation})
            glszm_df_sorted = glszm_df.sort_values(by=['correlation'], ascending=False)
            
            y_glszm = np.array(glszm_df_sorted['feature'])
            x_glszm = np.array(glszm_df_sorted['correlation'])
            glszm_mask = x_glszm > threshold
            glszm_mask_off = x_glszm < threshold
        
        if "gldm_features" in exclude_feature_groups:
            list_of_groups.remove(gldm_features)
        else:
            gldm_correlation = []
            for feature in gldm_features:
                gldm_correlation.append(correlation_df.loc[feature][0])
            gldm_clean = [substring.replace('original_gldm_', '') for substring in gldm_features]
            gldm_df= pd.DataFrame({"feature":gldm_clean, "correlation": gldm_correlation})
            gldm_df_sorted = gldm_df.sort_values(by=['correlation'], ascending=False)
            
            y_gldm = np.array(gldm_df_sorted['feature'])
            x_gldm = np.array(gldm_df_sorted['correlation'])
            gldm_mask = x_gldm > threshold
            gldm_mask_off = x_gldm < threshold
        
        if "log_features" in exclude_feature_groups:
            list_of_groups.remove(log_features)
        else:
            log_correlation = []
            for feature in log_features:
                log_correlation.append(correlation_df.loc[feature][0])
            log_clean = [substring.replace('log-sigma-', '') for substring in log_features]
            log_df= pd.DataFrame({"feature":log_clean, "correlation": log_correlation})
            log_df_sorted = log_df.sort_values(by=['correlation'], ascending=False)
            
            y_log = np.array(log_df_sorted['feature'])
            x_log = np.array(log_df_sorted['correlation'])
            log_mask = x_log > threshold
            log_mask_off = x_log < threshold
        
        if "wavelet_features" in exclude_feature_groups:
            list_of_groups.remove(wavelet_features)
        else:
            wavelet_correlation = []
            for feature in wavelet_features:
                wavelet_correlation.append(correlation_df.loc[feature][0])
            wavelet_clean = [substring.replace('wavelet-', '') for substring in wavelet_features]
            wavelet_df= pd.DataFrame({"feature":wavelet_clean, "correlation": wavelet_correlation})
            wavelet_df_sorted = wavelet_df.sort_values(by=['correlation'], ascending=False)
            
            y_wavelet = np.array(wavelet_df_sorted['feature'])
            x_wavelet = np.array(wavelet_df_sorted['correlation'])
            wavelet_mask = x_wavelet > threshold
            wavelet_mask_off = x_wavelet < threshold
        
        
        length = 0
        for nr in range(len(list_of_groups)):
            length = length + len(list_of_groups[nr])
        
        ratio_list = []
        for nr in range(len(list_of_groups)):
            ratio = len(list_of_groups[nr])/length
            ratio_list.append(ratio)
        
        fig, axis = plt.subplots(figsize=(8,(length/3.5)), nrows=8-len(exclude_feature_groups), ncols=1, gridspec_kw={"height_ratios": ratio_list})
        ct = 0
        
        axis[0].set_title('Correlation per feature')
        if "shape_features" not in exclude_feature_groups:
            axis[ct].barh(y_shape[shape_mask],x_shape[shape_mask], color = "green")
            axis[ct].barh(y_shape[shape_mask_off],x_shape[shape_mask_off], color = "gray")
            axis[ct].set_ylabel('Shape features')
            axis[ct].axvline(x=threshold,linewidth=1, color='k', ls='--')
            axis[ct].set_xlim([-0.05, 1.05])
            ct = ct + 1
        
        if "first_order_features" not in exclude_feature_groups:
            axis[ct].barh(y_first_order[first_order_mask],x_first_order[first_order_mask], color = "green")
            axis[ct].barh(y_first_order[first_order_mask_off],x_first_order[first_order_mask_off], color = "gray")
            axis[ct].set_ylabel('First order features')
            axis[ct].axvline(x=threshold,linewidth=1, color='k', ls='--')
            axis[ct].set_xlim([-0.05, 1.05])
            ct = ct + 1
        
        if "glcm_features" not in exclude_feature_groups:
            axis[ct].barh(y_glcm[glcm_mask],x_glcm[glcm_mask], color = "green")
            axis[ct].barh(y_glcm[glcm_mask_off],x_glcm[glcm_mask_off], color = "gray")
            axis[ct].set_ylabel('GLCM')
            axis[ct].axvline(x=threshold,linewidth=1, color='k', ls='--')
            axis[ct].set_xlim([-0.05, 1.05])
            ct = ct + 1
        
        if "glrlm_features" not in exclude_feature_groups:
            axis[ct].barh(y_glrlm[glrlm_mask],x_glrlm[glrlm_mask], color = "green")
            axis[ct].barh(y_glrlm[glrlm_mask_off],x_glrlm[glrlm_mask_off], color = "gray")
            axis[ct].set_ylabel('GLRLM')
            axis[ct].axvline(x=threshold,linewidth=1, color='k', ls='--')
            axis[ct].set_xlim([-0.05, 1.05])
            ct = ct + 1
        
        if "glszm_features" not in exclude_feature_groups:
            axis[ct].barh(y_glszm[glszm_mask],x_glszm[glszm_mask], color = "green")
            axis[ct].barh(y_glszm[glszm_mask_off],x_glszm[glszm_mask_off], color = "gray")
            axis[ct].set_ylabel('GLSZM')
            axis[ct].axvline(x=threshold,linewidth=1, color='k', ls='--')
            axis[ct].set_xlim([-0.05, 1.05])
            ct = ct + 1
            
        if "gldm_features" not in exclude_feature_groups:
            axis[ct].barh(y_gldm[gldm_mask],x_gldm[gldm_mask], color = "green")
            axis[ct].barh(y_gldm[gldm_mask_off],x_gldm[gldm_mask_off], color = "gray")
            axis[ct].set_ylabel('GLDM')
            axis[ct].axvline(x=threshold,linewidth=1, color='k', ls='--')
            axis[ct].set_xlim([-0.05, 1.05])
            ct = ct + 1
            
        if "log_features" not in exclude_feature_groups:
            axis[ct].barh(y_log[log_mask],x_log[log_mask], color = "green")
            axis[ct].barh(y_log[log_mask_off],x_log[log_mask_off], color = "gray")
            axis[ct].set_ylabel('LoG features')
            axis[ct].axvline(x=threshold,linewidth=1, color='k', ls='--')
            axis[ct].set_xlim([-0.05, 1.05])
            ct = ct + 1
        
        if "wavelet_features" not in exclude_feature_groups:
            axis[ct].barh(y_wavelet[wavelet_mask],x_wavelet[wavelet_mask], color = "green")
            axis[ct].barh(y_wavelet[wavelet_mask_off],x_wavelet[wavelet_mask_off], color = "gray")
            axis[ct].set_ylabel('Wavelet features')
            axis[ct].axvline(x=threshold,linewidth=1, color='k', ls='--')
            axis[ct].set_xlim([-0.05, 1.05])

        fig.subplots_adjust(hspace=0)    
