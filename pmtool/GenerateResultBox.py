# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:31:11 2022

@author: m.beuque
"""

import sklearn
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from ResultSet import ResultSet
import seaborn as sns


class GenerateResultBox(ResultSet):
    '''This module is inherited from ResultSet class and allows for results generation.'''

    def get_optimal_threshold(self):
        ##to obtain a good threshold based on the train dataset
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.train_df["labels"], self.train_df["predictions"])
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        self._threshold = optimal_threshold
        return optimal_threshold

    def _linking_data(self, label):
        if label == "train":
            y_label, y_pred = self.train_df["labels"], self.train_df["predictions"]
        if label == "test":
            y_label, y_pred = self.test_df["labels"], self.test_df["predictions"]
        if label == "external":
            y_label, y_pred = self.external_df["labels"], self.external_df["predictions"]
        return y_label, y_pred

    def get_results(self, label):  # better function below to get results with confidence interval
        ##optimal threshold: reuse the one computed on the train dataset
        ##label: index of the dataframe, can be "external radiomics results"
        ##returns a dataframe with auc accuracy precision recall f1-score
        y_label, y_pred = self._linking_data(label)
        dict_results = {}
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_label, y_pred)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        dict_results["auc"] = roc_auc
        if self._threshold == -1:
            self.get_optimal_threshold()
        y_pred_binary = (np.array(y_pred) > self.__threshold).astype(int)
        dict_results["balanced accuracy"] = [sklearn.metrics.balanced_accuracy_score(y_label, y_pred_binary)]
        dict_results["precision"] = [sklearn.metrics.precision_score(y_label, y_pred_binary)]
        dict_results["recall"] = [sklearn.metrics.recall_score(y_label, y_pred_binary)]
        dict_results["f1 score"] = [sklearn.metrics.f1_score(y_label, y_pred_binary)]
        df_results = pd.DataFrame.from_dict(dict_results)
        df_results = df_results.reset_index(drop=True)
        df_results.index = [label]
        return df_results

    def _bootstrap(self, label, pred, f, nsamples):
        stats = []
        for b in range(nsamples):
            random_list = np.random.randint(label.shape[0], size=label.shape[0])
            stats.append(f(label[random_list], pred[random_list]))
        return np.percentile(stats, (2.5, 97.5))

    def _get_ci(self, y_label, y_pred, f, nsamples):
        ci = self._bootstrap(y_label, y_pred, f, nsamples)
        return ["%0.2f CI [%0.2f,%0.2f]" % (f(y_label, y_pred), ci[0], ci[1])]  # doesn't compute the mean of the score

    def _get_ci_for_auc(self, y_label, y_pred, nsamples):
        auc_values = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        for b in range(nsamples):
            idx = np.random.randint(y_label.shape[0], size=y_label.shape[0])
            temp_pred = y_pred[idx]
            temp_fpr, temp_tpr, temp_thresholds = sklearn.metrics.roc_curve(y_label[idx], temp_pred)
            roc_auc = sklearn.metrics.auc(temp_fpr, temp_tpr)
            auc_values.append(roc_auc)
            interp_tpr = np.interp(mean_fpr, temp_fpr, temp_tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        ci_auc = np.percentile(auc_values, (2.5, 97.5))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_label, y_pred)
        return ["%0.2f CI [%0.2f,%0.2f]" % (sklearn.metrics.auc(fpr, tpr), ci_auc[0], ci_auc[1])]

    def get_stats_with_ci(self, label, nsamples=2000):
        ##optimal threshold: reuse the one computed on the train dataset
        ##label: index of the dataframe, can be "external radiomics results"
        ##returns a dataframe with auc accuracy precision recall f1-score
        dict_results = {}
        y_label, y_pred = self._linking_data(label)
        dict_results["auc"] = self._get_ci_for_auc(y_label, y_pred, nsamples)
        if self._threshold == -1:
            self.get_optimal_threshold()
        y_pred_binary = (np.array(y_pred) > self._threshold).astype(int)
        dict_results["accuracy"] = self._get_ci(y_label, y_pred_binary, sklearn.metrics.balanced_accuracy_score,
                                                nsamples)
        dict_results["precision"] = self._get_ci(y_label, y_pred_binary, sklearn.metrics.precision_score, nsamples)
        dict_results["recall"] = self._get_ci(y_label, y_pred_binary, sklearn.metrics.recall_score, nsamples)
        dict_results["f1 score"] = self._get_ci(y_label, y_pred_binary, sklearn.metrics.f1_score, nsamples)
        df_results = pd.DataFrame.from_dict(dict_results)
        df_results = df_results.reset_index(drop=True)
        df_results.index = [label]
        return df_results

    def _find_available_data(self):
        list_available_data = []
        if len(self.train_df) > 0:
            list_available_data.append("train")
        if len(self.test_df) > 0:
            list_available_data.append("test")
        if len(self.external_df) > 0:
            print(len(self.external_df))
            list_available_data.append("external")
        return list_available_data

    def plot_roc_auc_ci(self, title, nsamples=2000):

        curves_to_plot = self._find_available_data()

        if "train" in curves_to_plot:
            true_train_outcome, proba_train = self._linking_data("train")
            auc_values_train = []
            tprs_train = []
            mean_fpr_train = np.linspace(0, 1, 100)

            for b in range(nsamples):
                idx_train = np.random.randint(np.array(list(true_train_outcome)).shape[0],
                                              size=np.array(list(true_train_outcome)).shape[0])
                pred = proba_train[idx_train]
                fpr_train, tpr_train, thresholds_train = sklearn.metrics.roc_curve(
                    np.array(list(true_train_outcome))[idx_train], pred)
                interp_tpr_train = np.interp(mean_fpr_train, fpr_train, tpr_train)
                interp_tpr_train[0] = 0.0
                tprs_train.append(interp_tpr_train)
                roc_auc = sklearn.metrics.auc(fpr_train, tpr_train)
                auc_values_train.append(roc_auc)

            mean_tpr_train = np.mean(tprs_train, axis=0)
            mean_tpr_train[-1] = 1.0
            mean_auc_train = sklearn.metrics.auc(mean_fpr_train,
                                                 mean_tpr_train)  # does mean auc makes sense or should it be from the overall predictions?
            ci_auc_train = np.percentile(auc_values_train, (2.5, 97.5))

        if "external" in curves_to_plot:
            true_external_binary_outcome, proba_external = self._linking_data("external")
            auc_values_external = []
            tprs_external = []
            mean_fpr_external = np.linspace(0, 1, 100)

            for b in range(nsamples):
                idx_external = np.random.randint(np.array(list(true_external_binary_outcome)).shape[0],
                                                 size=np.array(list(true_external_binary_outcome)).shape[0])
                pred = proba_external[idx_external]
                fpr_external, tpr_external, thresholds_external = sklearn.metrics.roc_curve(
                    np.array(list(true_external_binary_outcome))[idx_external], pred)
                interp_tpr_external = np.interp(mean_fpr_external, fpr_external, tpr_external)
                interp_tpr_external[0] = 0.0
                tprs_external.append(interp_tpr_external)
                roc_auc = sklearn.metrics.auc(fpr_external, tpr_external)
                auc_values_external.append(roc_auc)

            mean_tpr_external = np.mean(tprs_external, axis=0)
            mean_tpr_external[-1] = 1.0
            mean_auc_external = sklearn.metrics.auc(mean_fpr_external,
                                                    mean_tpr_external)  # does mean auc makes sense or should it be from the overall predictions?
            ci_auc_external = np.percentile(auc_values_external, (2.5, 97.5))

        if "test" in curves_to_plot:
            true_test_outcome, proba_test = self._linking_data("test")
            auc_values = []
            tprs = []
            mean_fpr = np.linspace(0, 1, 100)

            for b in range(nsamples):
                idx_test = np.random.randint(np.array(list(true_test_outcome)).shape[0],
                                             size=np.array(list(true_test_outcome)).shape[0])
                pred = proba_test[idx_test]
                fpr_test, tpr_test, thresholds_test = sklearn.metrics.roc_curve(
                    np.array(list(true_test_outcome))[idx_test], pred)
                interp_tpr = np.interp(mean_fpr, fpr_test, tpr_test)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                roc_auc = sklearn.metrics.auc(fpr_test, tpr_test)
                auc_values.append(roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = sklearn.metrics.auc(mean_fpr,
                                           mean_tpr)  # does mean auc makes sense or should it be from the overall predictions?
            ci_auc = np.percentile(auc_values, (2.5, 97.5))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        if "train" in curves_to_plot:
            ax.plot(mean_fpr_train, mean_tpr_train, color='darkorange',
                    label=r'Mean ROC train (AUC = %0.2f CI [%0.2f,%0.2f])' % (
                    mean_auc_train, ci_auc_train[0], ci_auc_train[1]),
                    lw=2, alpha=.8)
            interval_tprs_train = np.array([np.percentile(np.array(tprs_train)[:, i], (2.5, 97.5)) for i in range(100)])
            tprs_upper_train = interval_tprs_train[:, 1]
            tprs_lower_train = interval_tprs_train[:, 0]
            ax.fill_between(mean_fpr_train, tprs_lower_train, tprs_upper_train, color='darkorange', alpha=.3)

        if "test" in curves_to_plot:
            ax.plot(mean_fpr, mean_tpr, color='blue',
                    label=r'Mean ROC test (AUC = %0.2f CI [%0.2f,%0.2f])' % (mean_auc, ci_auc[0], ci_auc[1]),
                    lw=2, alpha=.8)
            interval_tprs = np.array([np.percentile(np.array(tprs)[:, i], (2.5, 97.5)) for i in range(100)])
            tprs_upper = interval_tprs[:, 1]
            tprs_lower = interval_tprs[:, 0]
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='darkblue', alpha=.3)

        if "external" in curves_to_plot:
            ax.plot(mean_fpr_external, mean_tpr_external, color='green',
                    label=r'Mean ROC external (AUC = %0.2f CI [%0.2f,%0.2f])' % (
                    mean_auc_external, ci_auc_external[0], ci_auc_external[1]),
                    lw=2, alpha=.8)
            interval_tprs_external = np.array(
                [np.percentile(np.array(tprs_external)[:, i], (2.5, 97.5)) for i in range(100)])
            tprs_upper_external = interval_tprs_external[:, 1]
            tprs_lower_external = interval_tprs_external[:, 0]
            ax.fill_between(mean_fpr_external, tprs_lower_external, tprs_upper_external, color='green', alpha=.3)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.legend(loc="lower right")
        # plt.savefig(title+'.png',dpi=300)
        plt.show()
        return "done"

    def print_confusion_matrix(self, label, class_names, figsize=(6, 5), fontsize=14, normalize=True):
        sns.set(font_scale=1.4)
        if self._threshold == -1:
            self.get_optimal_threshold() #somehow doesn't update

        y_label, y_pred = self._linking_data(label)
        y_pred_binary = (y_pred > self._threshold).astype(int)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_label, y_pred_binary)
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,
        )
        fig = plt.figure(figsize=figsize)
        try:
            if normalize:
                fmt = '.2f'
                heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cmap="Blues", vmin=0, vmax=1)
            else:
                fmt = 'd'
                heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cmap="Blues", vmax=np.max(confusion_matrix), vmin=0)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        fig.tight_layout()
        # plt.savefig(figure_title,dpi=300)
        return fig


## test the class
if __name__ == '__main__':
    train_labels = [int(np.round(np.random.uniform(low=0, high=1))) for i in range(50)]
    train_predictions = [np.random.uniform(low=0, high=1) for i in range(50)]
    test_labels = [int(np.round(np.random.uniform(low=0, high=1))) for i in range(50)]
    test_predictions = [np.random.uniform(low=0, high=1) for i in range(50)]
    test = GenerateResultBox(train_labels=train_labels, train_predictions=train_predictions, test_labels=test_labels,
                             test_predictions=test_predictions)
    test.get_stats_with_ci("train")
    test.print_confusion_matrix("train",["0","1"])