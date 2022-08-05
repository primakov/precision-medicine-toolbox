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

    def normality_check(self, features_to_plot: list=[], p_thresh: float = 0.05) -> list:
        """
        Perform Shapiro-Wilcoxon normality check for all the features.

        Arguments:
            features_to_plot: List of specific features to be selected (otherwise selects all the numerical features).
            p_thresh: Shapiro-Wilk test p-value.

        Returns:
            List of the features with normal distribution.
        """

        if not features_to_plot:
            features_to_plot = self._feature_column

        num_features = []
        for feature in features_to_plot:
            if self._feature_dataframe[feature].dtype != 'object':
                num_features.append(feature)

        features_norm_distr = []
        for feature in num_features:
            shapiro_test = sp.stats.shapiro(np.array(list(self._feature_dataframe[feature]))).pvalue
            if shapiro_test > p_thresh:
                features_norm_distr.append(feature)

        return features_norm_distr


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

    def plot_correlation_matrix(self, features_to_plot: list = [], corr_method: str = 'spearman', save_to_csv: bool = False):
        """Plot correlation (Spearman's by default) matrix for the features into interactive .html report.

        Arguments:
            features_to_plot: List of specific features to be selected (otherwise selects all the numerical features).
            corr_method: Method of calculation: {'pearson', 'kendall', 'spearman'}.
            save_to_csv: Enable/disable saving correlation dataframe to .csv file.
        """

        pio.renderers.default = 'iframe'

        if not features_to_plot:
            features_to_plot = self._feature_column

        num_features = []
        for feature in features_to_plot:
            if self._feature_dataframe[feature].dtype != 'object':
                num_features.append(feature)

        corr_matrix_df = self._feature_dataframe[num_features].corr(method=corr_method)

        data = go.Heatmap(
            z=np.abs(np.array(corr_matrix_df)),
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

        if save_to_csv:
            corr_matrix_df.to_csv(os.path.splitext(self._feature_path)[0] + '_correlation_matrix.csv')

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
        """Calculate basic statistical scores (such as: number of missing values, mean, std, min, max, Mann-Whitney test p-values for binary classes, univariate ROC AUC for binary classes, Spearman's correlation with volume if volumetric feature name is sent to function, Shapiro-Wilk test p-values) for each feature and save it to .csv file.

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
        p_shapiro_test = []
        for feature in num_features:
            p_shapiro_test.append(sp.stats.shapiro(np.array(list(self._feature_dataframe[feature]))).pvalue)
        stats_dataframe = pd.concat([stats_dataframe,
                                     pd.DataFrame({'p_shapiro_test': p_shapiro_test}, index=num_features)],
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
        """Split all features into lists by feature type if you used Pyradiomics.

        Returns:
            Separate lists of original shape, first order, GLCM, GLRLM, GLSZM, and GLDM, as well as Laplasian of Gausian and wavelet features."""
        
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
