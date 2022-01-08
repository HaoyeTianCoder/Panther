import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn2_circles, venn3, venn3_circles
from keras.models import Sequential
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC, LinearSVC
import sklearn.metrics as metrics
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import *
from keras.layers import LSTM, Dense, Dropout

import tensorflow as tf
from keras.layers import Dense, Embedding, Dropout, Input, Concatenate

from experiment.deep_learning import *

class Prediction:
    def __init__(self, dataset, labels, record, feature1_length, algorithm, split_method, kfold=10, ):
        self.dataset = dataset
        self.record = record
        self.labels = labels
        self.algorithm = algorithm
        self.split_method = split_method
        self.kfold = kfold
        self.feature1_length = feature1_length

    def shap_value(self, xgb1, x_test_xgb1, xgb2, x_test_xgb2, xgb_all, x_test, learned, engineered, combine, patch_name_test):
        Bert_feature_name = ['B-'+str(i) for i in range(1024)]
        Bert_cross_feature_name = ['B-'+str(i) for i in range(2050)]
        CC2V_feature_name = ['C-'+str(i) for i in range(240)]
        CC2V_cross_feature_name = ['C-'+str(i) for i in range(482)]

        if x_test_xgb1.shape[1] == 1024:
            learned_feature_name = Bert_feature_name
        elif x_test_xgb1.shape[1] == 2050:
            learned_feature_name = Bert_cross_feature_name
        elif x_test_xgb1.shape[1] == 240:
            learned_feature_name = CC2V_feature_name
        elif x_test_xgb1.shape[1] == 482:
            learned_feature_name = CC2V_cross_feature_name
        else:
            raise

        if x_test_xgb2.shape[1] == 195:
            ODS_feature_name = ['P4J_FORMER_INSERT_CONTROL_RF', 'P4J_FORMER_INSERT_GUARD_RF', 'P4J_FORMER_INSERT_STMT_RF', 'P4J_FORMER_REPLACE_COND_RF', 'P4J_FORMER_REPLACE_STMT_RF', 'P4J_FORMER_REMOVE_PARTIAL_IF', 'P4J_FORMER_REMOVE_STMT', 'P4J_FORMER_REMOVE_WHOLE_IF', 'P4J_FORMER_REMOVE_WHOLE_BLOCK', 'P4J_FORMER_OP_ADD_AF', 'P4J_FORMER_OP_SUB_AF', 'P4J_FORMER_OP_MUL_AF', 'P4J_FORMER_OP_DIV_AF', 'P4J_FORMER_OP_MOD_AF', 'P4J_FORMER_OP_LE_AF', 'P4J_FORMER_OP_LT_AF', 'P4J_FORMER_OP_GE_AF', 'P4J_FORMER_OP_GT_AF', 'P4J_FORMER_OP_EQ_AF', 'P4J_FORMER_OP_NE_AF', 'P4J_FORMER_UOP_INC_AF', 'P4J_FORMER_UOP_DEC_AF', 'P4J_FORMER_ASSIGN_LHS_AF', 'P4J_FORMER_ASSIGN_ZERO_AF', 'P4J_FORMER_ASSIGN_CONST_AF', 'P4J_FORMER_CHANGED_AF', 'P4J_FORMER_DEREF_AF', 'P4J_FORMER_INDEX_AF', 'P4J_FORMER_MEMBER_ACCESS_AF', 'P4J_FORMER_CALLEE_AF', 'P4J_FORMER_CALL_ARGUMENT_AF', 'P4J_FORMER_ABST_V_AF', 'P4J_FORMER_STMT_LABEL_AF', 'P4J_FORMER_STMT_LOOP_AF', 'P4J_FORMER_STMT_ASSIGN_AF', 'P4J_FORMER_STMT_CALL_AF', 'P4J_FORMER_STMT_COND_AF', 'P4J_FORMER_STMT_CONTROL_AF', 'P4J_FORMER_R_STMT_ASSIGN_AF', 'P4J_FORMER_R_STMT_CALL_AF', 'P4J_FORMER_R_STMT_COND_AF', 'P4J_FORMER_R_STMT_CONTROL_AF', 'P4J_FORMER_MODIFIED_VF', 'P4J_FORMER_MODIFIED_SIMILAR_VF', 'P4J_FORMER_FUNC_ARGUMENT_VF', 'P4J_FORMER_MEMBER_VF', 'P4J_FORMER_LOCAL_VARIABLE_VF', 'P4J_FORMER_GLOBAL_VARIABLE_VF', 'P4J_FORMER_ZERO_CONST_VF', 'P4J_FORMER_NONZERO_CONST_VF', 'P4J_FORMER_STRING_LITERAL_VF', 'P4J_FORMER_SIZE_LITERAL_VF', 'P4J_LATER_INSERT_CONTROL_RF', 'P4J_LATER_INSERT_GUARD_RF', 'P4J_LATER_INSERT_STMT_RF', 'P4J_LATER_REPLACE_COND_RF', 'P4J_LATER_REPLACE_STMT_RF', 'P4J_LATER_REMOVE_PARTIAL_IF', 'P4J_LATER_REMOVE_STMT', 'P4J_LATER_REMOVE_WHOLE_IF', 'P4J_LATER_REMOVE_WHOLE_BLOCK', 'P4J_LATER_OP_ADD_AF', 'P4J_LATER_OP_SUB_AF', 'P4J_LATER_OP_MUL_AF', 'P4J_LATER_OP_DIV_AF', 'P4J_LATER_OP_MOD_AF', 'P4J_LATER_OP_LE_AF', 'P4J_LATER_OP_LT_AF', 'P4J_LATER_OP_GE_AF', 'P4J_LATER_OP_GT_AF', 'P4J_LATER_OP_EQ_AF', 'P4J_LATER_OP_NE_AF', 'P4J_LATER_UOP_INC_AF', 'P4J_LATER_UOP_DEC_AF', 'P4J_LATER_ASSIGN_LHS_AF', 'P4J_LATER_ASSIGN_ZERO_AF', 'P4J_LATER_ASSIGN_CONST_AF', 'P4J_LATER_CHANGED_AF', 'P4J_LATER_DEREF_AF', 'P4J_LATER_INDEX_AF', 'P4J_LATER_MEMBER_ACCESS_AF', 'P4J_LATER_CALLEE_AF', 'P4J_LATER_CALL_ARGUMENT_AF', 'P4J_LATER_ABST_V_AF', 'P4J_LATER_STMT_LABEL_AF', 'P4J_LATER_STMT_LOOP_AF', 'P4J_LATER_STMT_ASSIGN_AF', 'P4J_LATER_STMT_CALL_AF', 'P4J_LATER_STMT_COND_AF', 'P4J_LATER_STMT_CONTROL_AF', 'P4J_LATER_R_STMT_ASSIGN_AF', 'P4J_LATER_R_STMT_CALL_AF', 'P4J_LATER_R_STMT_COND_AF', 'P4J_LATER_R_STMT_CONTROL_AF', 'P4J_LATER_MODIFIED_VF', 'P4J_LATER_MODIFIED_SIMILAR_VF', 'P4J_LATER_FUNC_ARGUMENT_VF', 'P4J_LATER_MEMBER_VF', 'P4J_LATER_LOCAL_VARIABLE_VF', 'P4J_LATER_GLOBAL_VARIABLE_VF', 'P4J_LATER_ZERO_CONST_VF', 'P4J_LATER_NONZERO_CONST_VF', 'P4J_LATER_STRING_LITERAL_VF', 'P4J_LATER_SIZE_LITERAL_VF', 'P4J_SRC_INSERT_CONTROL_RF', 'P4J_SRC_INSERT_GUARD_RF', 'P4J_SRC_INSERT_STMT_RF', 'P4J_SRC_REPLACE_COND_RF', 'P4J_SRC_REPLACE_STMT_RF', 'P4J_SRC_REMOVE_PARTIAL_IF', 'P4J_SRC_REMOVE_STMT', 'P4J_SRC_REMOVE_WHOLE_IF', 'P4J_SRC_REMOVE_WHOLE_BLOCK', 'P4J_SRC_OP_ADD_AF', 'P4J_SRC_OP_SUB_AF', 'P4J_SRC_OP_MUL_AF', 'P4J_SRC_OP_DIV_AF', 'P4J_SRC_OP_MOD_AF', 'P4J_SRC_OP_LE_AF', 'P4J_SRC_OP_LT_AF', 'P4J_SRC_OP_GE_AF', 'P4J_SRC_OP_GT_AF', 'P4J_SRC_OP_EQ_AF', 'P4J_SRC_OP_NE_AF', 'P4J_SRC_UOP_INC_AF', 'P4J_SRC_UOP_DEC_AF', 'P4J_SRC_ASSIGN_LHS_AF', 'P4J_SRC_ASSIGN_ZERO_AF', 'P4J_SRC_ASSIGN_CONST_AF', 'P4J_SRC_CHANGED_AF', 'P4J_SRC_DEREF_AF', 'P4J_SRC_INDEX_AF', 'P4J_SRC_MEMBER_ACCESS_AF', 'P4J_SRC_CALLEE_AF', 'P4J_SRC_CALL_ARGUMENT_AF', 'P4J_SRC_ABST_V_AF', 'P4J_SRC_STMT_LABEL_AF', 'P4J_SRC_STMT_LOOP_AF', 'P4J_SRC_STMT_ASSIGN_AF', 'P4J_SRC_STMT_CALL_AF', 'P4J_SRC_STMT_COND_AF', 'P4J_SRC_STMT_CONTROL_AF', 'P4J_SRC_R_STMT_ASSIGN_AF', 'P4J_SRC_R_STMT_CALL_AF', 'P4J_SRC_R_STMT_COND_AF', 'P4J_SRC_R_STMT_CONTROL_AF', 'P4J_SRC_MODIFIED_VF', 'P4J_SRC_MODIFIED_SIMILAR_VF', 'P4J_SRC_FUNC_ARGUMENT_VF', 'P4J_SRC_MEMBER_VF', 'P4J_SRC_LOCAL_VARIABLE_VF', 'P4J_SRC_GLOBAL_VARIABLE_VF', 'P4J_SRC_ZERO_CONST_VF', 'P4J_SRC_NONZERO_CONST_VF', 'P4J_SRC_STRING_LITERAL_VF', 'P4J_SRC_SIZE_LITERAL_VF', 'codeMove', 'condBlockExcAdd', 'condBlockOthersAdd', 'condBlockRem', 'condBlockRetAdd', 'constChange', 'copyPaste', 'expArithMod', 'expLogicExpand', 'expLogicMod', 'expLogicReduce', 'missNullCheckN', 'missNullCheckP', 'notClassified', 'singleLine', 'unwrapIfElse', 'unwrapMethod', 'unwrapTryCatch', 'wrapsElse', 'wrapsIf', 'wrapsIfElse', 'wrapsLoop', 'wrapsMethod', 'wrapsTryCatch', 'wrongMethodRef', 'wrongVarRef', 'UpdateLiteral', 'addLineNo', 'addThis', 'condLogicReduce', 'dupArgsInvocation', 'ifTrue', 'insertBooleanLiteral', 'insertIfFalse', 'insertNewConstLiteral', 'patchedFileNo', 'removeNullinCond', 'rmLineNo', 'updIfFalse']
        else:
            raise

        # xgb_1 SHAP
        explainer_tree1 = shap.TreeExplainer(xgb1,)
        x_test_xgb1 = pd.DataFrame(x_test_xgb1, columns=learned_feature_name)
        shap_values_tree1 = explainer_tree1.shap_values(x_test_xgb1, )
        # plt.subplot(1, 2, 2)
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
        shap.summary_plot(shap_values_tree1, x_test_xgb1, max_display=10, plot_size=(10, 5))
        plt.savefig('../images/xgb1_shap.png')
        plt.clf()
        # plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
        # shap.summary_plot(shap_values_tree1, x_test_xgb1, plot_type="bar", max_display=10)
        # plt.savefig('../images/xgb1_imp.png')
        # plt.clf()

        # xgb_2 SHAP
        explainer_tree2 = shap.TreeExplainer(xgb2,)
        x_test_xgb2 = pd.DataFrame(x_test_xgb2, columns=ODS_feature_name)
        shap_values_tree2 = explainer_tree2.shap_values(x_test_xgb2, )
        # plt.subplot(1, 2, 2)
        plt.subplots_adjust(left=0.417, right=0.97, top=0.9, bottom=0.1)
        shap.summary_plot(shap_values_tree2, x_test_xgb2, max_display=10, plot_size=(10, 5))
        plt.savefig('../images/xgb2_shap.png')
        plt.clf()
        # plt.subplots_adjust(left=0.417, right=0.97, top=0.9, bottom=0.1)
        # shap.summary_plot(shap_values_tree2, x_test_xgb2, plot_type="bar", max_display=10)
        # plt.savefig('../images/xgb2_imp.png')
        # plt.clf()

        # xgb_all SHAP
        explainer_tree_all = shap.TreeExplainer(xgb_all,)
        x_test = pd.DataFrame(x_test, columns=learned_feature_name+ODS_feature_name)
        shap_values_tree = explainer_tree_all.shap_values(x_test, )
        plt.subplots_adjust(left=0.417, right=0.97, top=0.9, bottom=0.1)
        shap.summary_plot(shap_values_tree, x_test, max_display=10, plot_size=(10, 5))
        plt.savefig('../images/xgb_all_shap.png')
        plt.clf()
        # plt.subplots_adjust(left=0.417, right=0.97, top=0.9, bottom=0.1)
        # shap.summary_plot(shap_values_tree, x_test, plot_type="bar", max_display=10, plot_size=(10, 5))
        # plt.savefig('../images/xgb_all_imp.png')
        # plt.clf()


        # interaction
        shap_interaction_values = explainer_tree_all.shap_interaction_values(x_test)
        shap.summary_plot(shap_interaction_values, x_test, max_display=10)
        # only two features interact, no any other features involved.
        shap.dependence_plot(("singleLine", "B-1237"), shap_interaction_values, x_test, display_features=x_test)
        plt.savefig('../images/interaction_single.png')
        # except for 'L-1187', the final interaction result of 'singleLine' in figure is influenced by other features .
        # shap.dependence_plot('singleLine', shap_values_tree, x_test, interaction_index='L-1187',)


        # analysis single sample for the last time
        try:
            self.sample_analysis(learned, engineered, combine, patch_name_test, x_test, x_test_xgb1, x_test_xgb2, shap_values_tree, shap_values_tree1, shap_values_tree2, explainer_tree_all, explainer_tree1, explainer_tree2)
        except Exception as e:
            print('not this time')

    def sample_analysis(self, learned, engineered, combine, patch_name_test, x_test, x_test_xgb1, x_test_xgb2, shap_values_tree, shap_values_tree1, shap_values_tree2, explainer_tree, explainer_tree1, explainer_tree2):
        learned_set = set(learned)
        engineered_set = set(engineered)
        combine_set = set(combine)

        independent_learned = learned_set - engineered_set - combine_set
        independent_engineered = engineered_set - learned_set - combine_set
        independent_combine = combine_set - learned_set - engineered_set

        patch_sample = ''

        # learned
        patch_name = independent_learned.pop()
        index = list(patch_name_test).index(patch_name)
        patch_sample += 'patch identified by learned: {}\n'.format(patch_name)
        shap.initjs()
        shap4both = shap.force_plot(explainer_tree1.expected_value, shap_values_tree1[index], x_test_xgb1.loc[index])
        with open("../SHAP/shap4learned.html", "w+") as file:
            file.write(shap4both.html())

        # engineered
        patch_name = independent_engineered.pop()
        index = list(patch_name_test).index(patch_name)
        patch_sample += 'patch identified by engineered: {}\n'.format(patch_name)
        shap.initjs()
        # shap4one = shap.force_plot(explainer_tree.expected_value, shap_values_tree[index][2050:], x_test.loc[index][2050:])
        shap4both = shap.force_plot(explainer_tree2.expected_value, shap_values_tree2[index], x_test_xgb2.loc[index])
        with open("../SHAP/shap4engineered.html", "w+") as file:
            file.write(shap4both.html())

        # combine
        patch_name = independent_combine.pop()
        index = list(patch_name_test).index(patch_name)
        patch_sample += 'patch identified by combined: {}\n'.format(patch_name)
        shap.initjs()
        shap4both = shap.force_plot(explainer_tree.expected_value, shap_values_tree[index], x_test.loc[index])
        with open("../SHAP/shap4combine.html", "w+") as file:
            file.write(shap4both.html())

        # save patch record
        with open("../SHAP/patch_name.txt","w+") as file:
            file.write(patch_sample)

        print('success!')

        # from IPython.display import display, HTML, Image
        # display(shap4both)

    def independent_case(self, learned_all, engineered_all, combine_all):
        learned_set = set(learned_all)
        engineered_set = set(engineered_all)
        combine_set = set(combine_all)

        independent_learned = learned_set - engineered_set - combine_set
        independent_engineered = engineered_set - learned_set - combine_set
        independent_combine = combine_set - learned_set - engineered_set

        return independent_learned, independent_engineered, independent_combine

    def draw_venn2(self, learned_correctness_all, combine_correctness_all, learned_incorrectness_all, combine_incorrectness_all):
        # correct
        my_dpi = 150
        plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
        a = venn2(subsets=[set(learned_correctness_all), set(combine_correctness_all)],
                  set_labels=('learned', 'other'),
                  set_colors=("#01a2d9", "#c72e29"),
                  alpha=0.3,  # 透明度
                  normalize_to=0.6,  # venn图占据figure的比例，1.0为占满
                  )
        plt.title('Correct')
        plt.show()

        # incorrect
        my_dpi = 150
        plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
        b = venn2(subsets=[set(learned_incorrectness_all), set(combine_incorrectness_all)],
                  set_labels=('learned', 'other'),
                  set_colors=("#01a2d9", "#c72e29"),
                  alpha=0.3,  # 透明度
                  normalize_to=0.6,  # venn图占据figure的比例，1.0为占满
                  )
        plt.title('Incorrect')
        plt.show()

    def draw_venn3(self, learned_correctness_all, engineered_correctness_all, combine_correctness_all, learned_incorrectness_all, \
               engineered_incorrectness_all, combine_incorrectness_all):
        # correct
        my_dpi = 150
        plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
        a = venn3(subsets=[set(learned_correctness_all), set(engineered_correctness_all),set(combine_correctness_all)],
                  set_labels=('learned', 'engineered', 'combine'),
                  set_colors=("#01a2d9", "#31A354", "#c72e29"),
                  alpha=0.3,  # 透明度
                  normalize_to=0.6,  # venn图占据figure的比例，1.0为占满
                  )
        plt.title('Correct')
        plt.show()

        # incorrect
        my_dpi = 150
        plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
        b = venn3(subsets=[set(learned_incorrectness_all), set(engineered_incorrectness_all), set(combine_incorrectness_all)],
                  set_labels=('learned', 'engineered', 'combine'),
                  set_colors=("#01a2d9", "#31A354", "#c72e29"),
                  alpha=0.3,  # 透明度
                  normalize_to=0.6,  # venn图占据figure的比例，1.0为占满
                  )
        plt.title('Incorrect')
        plt.show()


    def venn_3(self, y_pred_lr_single, y_pred_xgb_single, y_pred_combine, y_test, patch_name_test):
        y_pred_lr_single = list(y_pred_lr_single)
        y_pred_xgb_single = list(y_pred_xgb_single)
        y_pred_combine = list(y_pred_combine)

        # correct patch
        index_p = [i for i,v in enumerate(y_test) if v == 1]
        learned_single_patch = [patch_name_test[i] for i in index_p if y_pred_lr_single[i] >= 0.5]
        engineered_single_patch = [patch_name_test[i] for i in index_p if y_pred_xgb_single[i] >= 0.5]
        combine_patch = [patch_name_test[i] for i in index_p if y_pred_combine[i] >= 0.5]
        my_dpi = 150
        plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
        g = venn3(subsets=[set(learned_single_patch), set(engineered_single_patch), set(combine_patch)],
                  set_labels=('learned', 'engineered', 'combine'),
                  set_colors=("#01a2d9", "#31A354", "#c72e29"),
                  alpha=0.8,  # 透明度
                  normalize_to=0.6,  # venn图占据figure的比例，1.0为占满
                  )
        plt.show()

        # incorrect patch
        index_n = [i for i,v in enumerate(y_test) if v == 0]
        learned_single_patch = [patch_name_test[i] for i in index_n if y_pred_lr_single[i] < 0.5]
        engineered_single_patch = [patch_name_test[i] for i in index_n if y_pred_xgb_single[i] < 0.5]
        combine_patch = [patch_name_test[i] for i in index_n if y_pred_combine[i] < 0.5]
        my_dpi = 150
        plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
        g = venn3(subsets=[set(learned_single_patch), set(engineered_single_patch), set(combine_patch)],
                  set_labels=('learned', 'engineered', 'combine'),
                  set_colors=("#01a2d9", "#31A354", "#c72e29"),
                  alpha=0.8,  # 透明度
                  normalize_to=0.6,  # venn图占据figure的比例，1.0为占满
                  )
        plt.show()

        for k,v in enumerate(engineered_single_patch):
            if v not in learned_single_patch and v not in combine_patch:
                i = list(patch_name_test).index(v)
                print('lr prob: {}'.format(y_pred_lr_single[i]))
                print('xgb prob: {}'.format(y_pred_xgb_single[i]))
                print('combine prob: {}'.format(y_pred_combine[i]))
                print('name: {}'.format(v))

                print('')

        print('no no no')

    def venn_2(self, y_pred_lr_single, y_pred_combine, y_test, patch_name_test):
        y_pred_lr_single = list(y_pred_lr_single)
        y_pred_combine = list(y_pred_combine)

        # correct patch
        index_p = [i for i,v in enumerate(y_test) if v == 1]
        learned_single_patch = [patch_name_test[i] for i in index_p if y_pred_lr_single[i] >= 0.5]
        combine_patch = [patch_name_test[i] for i in index_p if y_pred_combine[i] >= 0.5]
        my_dpi = 150
        plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
        g = venn2(subsets=[set(learned_single_patch), set(combine_patch)],
                  set_labels=('learned', 'combine'),
                  set_colors=("#01a2d9", "#c72e29"),
                  alpha=0.8,  # 透明度
                  normalize_to=0.6,  # venn图占据figure的比例，1.0为占满
                  )
        plt.show()
        diff_correct = len(combine_patch)-len(learned_single_patch)
        p1 = [p for p in learned_single_patch if p not in combine_patch]

        # incorrect patch
        index_n = [i for i,v in enumerate(y_test) if v == 0]
        learned_single_patch = [patch_name_test[i] for i in index_n if y_pred_lr_single[i] < 0.5]
        combine_patch = [patch_name_test[i] for i in index_n if y_pred_combine[i] < 0.5]
        my_dpi = 150
        plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
        g = venn2(subsets=[set(learned_single_patch), set(combine_patch)],
                  set_labels=('learned', 'combine'),
                  set_colors=("#01a2d9", "#c72e29"),
                  alpha=0.8,  # 透明度
                  normalize_to=0.6,  # venn图占据figure的比例，1.0为占满
                  )
        plt.show()
        diff_incorrect = len(combine_patch)-len(learned_single_patch)
        # p2 = [p+'-P' for p in learned_single_patch if p not in combine_patch]
        p2 = []

        return diff_correct, diff_incorrect, p1+p2

    def predict_result(self, y_pred_lr_single, y_pred_xgb_single, y_pred_combine, y_test, patch_name_test):
        y_pred_lr_single = list(y_pred_lr_single)
        y_pred_combine = list(y_pred_combine)

        # correct patch
        index_p = [i for i,v in enumerate(y_test) if v == 1]
        learned_correctness = [patch_name_test[i] for i in index_p if y_pred_lr_single[i] >= 0.5]
        engineered_correctness = [patch_name_test[i] for i in index_p if y_pred_xgb_single[i] >= 0.5]
        combine_correctness = [patch_name_test[i] for i in index_p if y_pred_combine[i] >= 0.5]

        # incorrect patch
        index_n = [i for i,v in enumerate(y_test) if v == 0]
        learned_incorrectness = [patch_name_test[i] for i in index_n if y_pred_lr_single[i] < 0.5]
        engineered_incorrectness = [patch_name_test[i] for i in index_n if y_pred_xgb_single[i] < 0.5]
        combine_incorrectness = [patch_name_test[i] for i in index_n if y_pred_combine[i] < 0.5]

        return learned_correctness, engineered_correctness, combine_correctness, learned_incorrectness, engineered_incorrectness, combine_incorrectness

    def confusion_matrix(self, y_pred, y_test):
        for i in range(1, 10):
            y_pred_tn = [1 if p >= i / 10.0 else 0 for p in y_pred]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
            print('i:{}'.format(i / 10), end=' ')
            print('TP: %d -- TN: %d -- FP: %d -- FN: %d' % (tp, tn, fp, fn))

    def draw_auc(self, y_pred, y_test):
        # calculate the fpr and tpr for all thresholds of the classification
        preds = y_pred
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)

        # method I: plt
        plt.rcParams.update({'font.size': 15})
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc,color='black',)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--',color='black',)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def evaluation_metrics(self, y_true, y_pred_prob):
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        prc = precision_score(y_true=y_true, y_pred=y_pred)
        rc = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = 2 * prc * rc / (prc + rc)

        print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(y_true), y_true.count(0), y_true.count(1)))

        # minn = 1
        # for i in range(len(y_true)):
        #     if y_true[i] == 1:
        #         if y_pred_prob[i] < minn:
        #             minn = y_pred_prob[i]
        # print(minn)
        # y_pred = [1 if p >= (minn-0.000001) else 0 for p in y_pred_prob]
        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # recall_p = tp / (tp + fn)
        # recall_n = tn / (tn + fp)
        # print('Thresh: {:.2f}, AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(minn, auc_, recall_p, recall_n))
        # for i in range(0,1001):
        #     threshold = i*0.001
        #     y_pred = [1 if p >= threshold else 0 for p in y_pred_prob]
        #     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #     recall_p = tp / (tp + fn)
        #     recall_n = tn / (tn + fp)
        #     print('Thresh: {:.2f}, AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(threshold, auc_, recall_p, recall_n))


        print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
        # return , auc_
        return auc_, recall_p, recall_n, acc, prc, rc, f1

    def run_compare(self, ):

        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()

        dataset, labels, record = self.dataset, self.labels, self.record
        x_train, y_train, x_test, y_test = [], [], [], []

        # patchsim test data
        patchsim_name, y_patchsim = [], []
        with open('patchsim_result.txt', 'r+') as f:
            for line in f:
                name = line.split(',')[0][:-2]
                label = int(line.split(',')[1])
                patchsim_name.append(name)
                y_patchsim.append(label)
        for i in range(len(self.record)):
            patchName = self.record.iloc[i][1][2:]
            if patchName in patchsim_name:
                # as test data
                x_test.append(self.dataset[i])
                y_test.append(self.labels[i])
            else:
                # as train data
                x_train.append(self.dataset[i])
                y_train.append(self.labels[i])

        # standard data
        scaler = StandardScaler().fit(x_train)
        # scaler = MinMaxScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        y_train, y_test = np.array(y_train), np.array(y_test)

        clf = None
        xgb_params = {'objective': 'binary:logistic', 'verbosity': 0}
        if self.algorithm == 'lr':
            clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train, y=y_train)
        elif self.algorithm == 'lr_combine':
            x_train_lr = x_train[:, :self.feature1_length]
            lr = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train_lr, y=y_train)

            x_train_xgb2 = x_train[:, self.feature1_length:]
            x_train_xgb2_dmatrix = xgb.DMatrix(x_train_xgb2, label=y_train)
            xgb2 = xgb.train(params=xgb_params, dtrain=x_train_xgb2_dmatrix, num_boost_round=100)

            clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
        elif self.algorithm == 'rf':
            clf = RandomForestClassifier(n_estimators=100, ).fit(X=x_train, y=y_train)
        elif self.algorithm == 'xgb':
            dtrain = xgb.DMatrix(x_train, label=y_train)
            clf = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=100)
        elif self.algorithm == 'dnn':
            if x_train.shape[1] == 1024 or x_train.shape[1] == 2050:
                dnn_model = get_dnn(dimension=x_train.shape[1])
            elif x_train.shape[1] == 195 or x_train.shape[1] == 4510:
                dnn_model = get_dnn_4engineered(dimension=x_train.shape[1])
            # bert, CC2Vec, Doc2Vec
            else:
                dnn_model = get_dnn(dimension=x_train.shape[1])

            callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=1, mode="max", verbose=1), ]
            dnn_model.fit(x_train, y_train, callbacks=callback, batch_size=32, epochs=10, )

        # prediction
        if self.algorithm == 'lgb':
            x_test_lgb = x_test
            y_pred = clf.predict(x_test_lgb)
        elif self.algorithm == 'xgb':
            x_test_xgb = x_test
            x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)
            y_pred = clf.predict(x_test_xgb_dmatrix)
        elif self.algorithm == 'dnn':
            y_pred = dnn_model.predict(x_test)[:, 0]
        else:
            y_pred = clf.predict_proba(x_test)[:, 1]

        # for i in range(1, 10):
        #     y_pred_tn = [1 if p >= i / 10.0 else 0 for p in y_pred]
        #     tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
        #     print('i:{}'.format(i / 10), end=' ')
        #     print('TP: %d -- TN: %d -- FP: %d -- FN: %d' % (tp, tn, fp, fn))

        auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=list(y_test), y_pred_prob=list(y_pred))
        # print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (
        #     np.array(acc).mean(), np.array(prc).mean(), np.array(rc).mean(), np.array(f1).mean(),
        #     np.array(auc_).mean()))

        output2 = '------------------------------------\n'
        output2 += 'Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (
            np.array(acc).mean(), np.array(prc).mean(), np.array(rc).mean(), np.array(f1).mean(),
            np.array(auc_).mean())

        y_pred_int = [1 if p >= 0.5 else 0 for p in y_pred]
        index_c = [i for i in range(len(list(y_test))) if list(y_test)[i] == 1]
        index_c_learned = [str(i) for i in index_c if y_pred_int[i] == 1]
        index_c_patchsim = [str(i) for i in index_c if y_patchsim[i] == 1]

        index_in = [i for i in range(len(list(y_test))) if list(y_test)[i] == 0]
        index_in_learned = [str(i) for i in index_in if y_pred_int[i] == 0]
        index_in_patchsim = [str(i) for i in index_in if y_patchsim[i] == 0]
        self.draw_venn2(index_c_learned, index_c_patchsim, index_in_learned, index_in_patchsim)


        # import sklearn.metrics as metrics
        # # calculate the fpr and tpr for all thresholds of the classification
        # preds = y_pred
        # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        # roc_auc = metrics.auc(fpr, tpr)

        return output2

    def run_test_patchsim(self, fea_used):

        path_testdata_vector = '../data_vector_TestData_Bert'

        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()
        learned_correctness_all, engineered_correctness_all, combine_correctness_all, learned_incorrectness_all, engineered_incorrectness_all, combine_incorrectness_all = list(), list(), list(), list(), list(), list()

        if not os.path.exists(path=path_testdata_vector):
            # logging.info('miss the path of the test data ......')
            raise Exception('miss the path of the datset: {} ......'.format(path_testdata_vector))

        test_data = None
        if fea_used == 'learned':
            test_data = np.load(path_testdata_vector + '/learned_Bert.npy')
        elif fea_used == 'engineered':
            test_data = np.load(path_testdata_vector + '/engineered_ods.npy', allow_pickle=True)
        elif fea_used == 'combine':
            learned_feature = np.load(path_testdata_vector + '/learned_Bert.npy')
            engineered_feature = np.load(path_testdata_vector + '/engineered_ods.npy', allow_pickle=True)
            test_data = np.concatenate((learned_feature, engineered_feature), axis=1)
            # self.feature1_length = learned_feature.shape[1]
        else:
            raise
        y_test = np.load(path_testdata_vector + '/labels.npy')
        y_train = self.labels

        # avoid the patches of testing data from appearing in the training data
        x_train_list = self.dataset.tolist()
        x_test_list = test_data.tolist()
        del_index = []
        for i in range(len(x_train_list)):
            train = x_train_list[i]
            if train in x_test_list:
                del_index.append(i)
        x_train = np.delete(self.dataset, del_index, axis=0)
        y_train = np.delete(y_train, del_index, axis=0)

        # standard data
        scaler = StandardScaler().fit(x_train)
        # scaler = MinMaxScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(test_data)

        clf = None
        xgb_params = {'objective': 'binary:logistic', 'verbosity': 0}
        if self.algorithm == 'lr':
            clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train, y=y_train)
        elif self.algorithm == 'lr_combine':
            x_train_lr = x_train[:, :self.feature1_length]
            lr = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train_lr, y=y_train)

            x_train_xgb2 = x_train[:, self.feature1_length:]
            x_train_xgb2_dmatrix = xgb.DMatrix(x_train_xgb2, label=y_train)
            xgb2 = xgb.train(params=xgb_params, dtrain=x_train_xgb2_dmatrix, num_boost_round=100)

            clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
        elif self.algorithm == 'rf':
            clf = RandomForestClassifier(n_estimators=100, ).fit(X=x_train, y=y_train)
        elif self.algorithm == 'xgb':
            dtrain = xgb.DMatrix(x_train, label=y_train)
            clf = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=100)

        # prediction
        if self.algorithm == 'lgb':
            x_test_lgb = x_test
            y_pred = clf.predict(x_test_lgb)
        elif self.algorithm == 'xgb':
            x_test_xgb = x_test
            x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)
            y_pred = clf.predict(x_test_xgb_dmatrix)
        else:
            y_pred = clf.predict_proba(x_test)[:, 1]

        for i in range(1, 10):
            y_pred_tn = [1 if p >= i / 10.0 else 0 for p in y_pred]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
            print('i:{}'.format(i / 10), end=' ')
            print('TP: %d -- TN: %d -- FP: %d -- FN: %d' % (tp, tn, fp, fn))

        auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)
        # print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (
        #     np.array(acc).mean(), np.array(prc).mean(), np.array(rc).mean(), np.array(f1).mean(),
        #     np.array(auc_).mean()))

        output2 = '------------------------------------\n'
        output2 += 'Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (
            np.array(acc).mean(), np.array(prc).mean(), np.array(rc).mean(), np.array(f1).mean(),
            np.array(auc_).mean())


        import sklearn.metrics as metrics
        # calculate the fpr and tpr for all thresholds of the classification
        preds = y_pred
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)

        return output2
    def run_cvfold(self, ):

        # scaler = StandardScaler()
        # scaler.fit_transform(self.dataset)

        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()
        learned_correctness_all, engineered_correctness_all, combine_correctness_all, learned_incorrectness_all, engineered_incorrectness_all, combine_incorrectness_all = list(), list(), list(), list(), list(), list()

        for train_index, test_index in skf.split(self.dataset, self.labels):
            x_train, y_train = self.dataset[train_index], self.labels[train_index]
            x_test, y_test = self.dataset[test_index], self.labels[test_index]
            patch_name_test = np.array(self.record.loc[test_index]['name'])

            # standard data
            scaler = StandardScaler().fit(x_train)
            # scaler = MinMaxScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            xgb_params = {'objective': 'binary:logistic', 'verbosity': 0}

            clf = None
            if self.algorithm == 'lr':
                clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
            elif self.algorithm == 'lr_combine':
                x_train_lr = x_train[:,:self.feature1_length]
                lr = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train_lr, y=y_train)

                x_train_xgb2 = x_train[:,self.feature1_length:]
                x_train_xgb2_dmatrix = xgb.DMatrix(x_train_xgb2, label=y_train)
                xgb2 = xgb.train(params=xgb_params, dtrain=x_train_xgb2_dmatrix, num_boost_round=100)

                lr_all = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
            elif self.algorithm == 'dt':
                clf = DecisionTreeClassifier().fit(X=x_train, y=y_train, sample_weight=None)
            elif self.algorithm == 'rf':
                clf = RandomForestClassifier(n_estimators=100, ).fit(X=x_train, y=y_train)
            elif self.algorithm == 'xgb':
                dtrain = xgb.DMatrix(x_train, label=y_train)
                clf = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=100)
            elif self.algorithm == 'xgb_combine':
                x_train_xgb_all_dmatrix = xgb.DMatrix(x_train, label=y_train)

                x_train_xgb1 = x_train[:,:self.feature1_length]
                x_train_xgb1_dmatrix = xgb.DMatrix(x_train_xgb1, label=y_train)

                x_train_xgb2 = x_train[:,self.feature1_length:]
                x_train_xgb2_dmatrix = xgb.DMatrix(x_train_xgb2, label=y_train)

                xgb1 = xgb.train(params=xgb_params, dtrain=x_train_xgb1_dmatrix, num_boost_round=100)
                xgb2 = xgb.train(params=xgb_params, dtrain=x_train_xgb2_dmatrix, num_boost_round=100)
                xgb_all = xgb.train(params=xgb_params, dtrain=x_train_xgb_all_dmatrix, num_boost_round=100)
            elif self.algorithm == 'rf_combine':
                x_train_rf1 = x_train[:,:self.feature1_length]
                x_train_rf2 = x_train[:,self.feature1_length:]

                rf1 = RandomForestClassifier(n_estimators=100, ).fit(X=x_train_rf1, y=y_train)
                rf2 = RandomForestClassifier(n_estimators=100, ).fit(X=x_train_rf2, y=y_train)
                rf_all = RandomForestClassifier(n_estimators=100, ).fit(X=x_train, y=y_train)
            elif self.algorithm == 'lgb':
                x_train_lgb_dmatrix = lgb.Dataset(data=x_train, label=y_train)
                params = {'objective': 'binary', }
                num_boost_round = 100
                clf = lgb.train(params=params, train_set=x_train_lgb_dmatrix, num_boost_round=num_boost_round,)
            elif self.algorithm == 'nb':
                clf = GaussianNB().fit(X=x_train, y=y_train)
            elif self.algorithm == 'dnn':
                if x_train.shape[1] == 1024 or x_train.shape[1] == 2050:
                    dnn_model = get_dnn(dimension=x_train.shape[1])
                elif x_train.shape[1] == 195 or x_train.shape[1] == 4510:
                    dnn_model = get_dnn_4engineered(dimension=x_train.shape[1])
                # bert, CC2Vec, Doc2Vec
                else:
                    dnn_model = get_dnn(dimension=x_train.shape[1])

                callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=1, mode="max", verbose=1),]
                dnn_model.fit(x_train, y_train, callbacks=callback, batch_size=32, epochs=10, )
            elif self.algorithm == 'dnn_dnn':
                x_train_learned = x_train[:,:self.feature1_length]
                x_train_static = x_train[:,self.feature1_length:]
                callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=1, mode="max", verbose=1), ]

                combine_deep_model = get_dnn_dnn(x_train_learned.shape[1], x_train_static.shape[1])
                combine_deep_model.fit([x_train_learned, x_train_static], y_train, callbacks=callback, batch_size=32,
                          epochs=10, )
            elif self.algorithm == 'dnn_dnn_venn':
                x_train_lr = x_train[:,:self.feature1_length]
                x_train_xgb = x_train[:,self.feature1_length:]
                callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=1, mode="max", verbose=1), ]


                dnn_model1 = get_dnn(dimension=x_train_lr.shape[1])
                dnn_model1.fit(x_train_lr, y_train, callbacks=callback, batch_size=32, epochs=10, )

                dnn_model2 = get_dnn_4engineered(dimension=x_train_xgb.shape[1])
                dnn_model2.fit(x_train_xgb, y_train, callbacks=callback, batch_size=32, epochs=10, )

                combine_deep_model = get_dnn_dnn(x_train_lr.shape[1], x_train_xgb.shape[1])
                combine_deep_model.fit([x_train_lr, x_train_xgb], y_train, callbacks=callback, batch_size=32,
                          epochs=10, )

            elif self.algorithm == 'xgb_xgb':
                x_train_xgb1 = x_train[:,:self.feature1_length]
                x_train_xgb1_dmatrix = xgb.DMatrix(x_train_xgb1, label=y_train)

                x_train_xgb2 = x_train[:,self.feature1_length:]
                x_train_xgb2_dmatrix = xgb.DMatrix(x_train_xgb2, label=y_train)

                xgb1 = xgb.train(params=xgb_params, dtrain=x_train_xgb1_dmatrix, num_boost_round=100)
                xgb2 = xgb.train(params=xgb_params, dtrain=x_train_xgb2_dmatrix, num_boost_round=100)
            elif self.algorithm == 'rf_rf':
                x_train_rf1 = x_train[:,:self.feature1_length]
                x_train_rf2 = x_train[:,self.feature1_length:]

                rf1 = RandomForestClassifier(n_estimators=100, ).fit(X=x_train_rf1, y=y_train)
                rf2 = RandomForestClassifier(n_estimators=100, ).fit(X=x_train_rf2, y=y_train)

            elif self.algorithm == 'lr_rf':
                x_train_lr = x_train[:,:self.feature1_length]
                x_train_rf = x_train[:,self.feature1_length:]

                lr = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train_lr, y=y_train)
                rf = RandomForestClassifier().fit(X=x_train_rf, y=y_train)
            elif self.algorithm == 'wide_deep':
                x_train_lr = x_train[:,:self.feature1_length]
                x_train_xgb = x_train[:,self.feature1_length:]
                combine_deep_model = get_wide_deep(x_train_lr.shape[1], x_train_xgb.shape[1])
                callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=1, mode="max", verbose=1), ]
                combine_deep_model.fit([x_train_lr, x_train_xgb], y_train, callbacks=callback, batch_size=32,
                          epochs=10, )
            elif self.algorithm == 'lr_xgb':
                x_train_lr = x_train[:,:self.feature1_length]
                x_train_xgb = x_train[:,self.feature1_length:]

                lr = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train_lr, y=y_train)

                x_train_xgb_dmatrix = xgb.DMatrix(x_train_xgb, label=y_train)
                xgb_model = xgb.train(params=xgb_params, dtrain=x_train_xgb_dmatrix, num_boost_round=100)


            elif self.algorithm == 'lr_lgb':
                x_train_lr = x_train[:, :self.feature1_length]
                x_train_lgb = x_train[:, self.feature1_length:]

                lr = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train_lr, y=y_train)

                x_train_lgb_dmatrix = lgb.Dataset(data=x_train_lgb, label=y_train)

                params = {'objective': 'binary', 'verbosity': 0}
                num_boost_round = 100
                lgb_model = lgb.train(params=params, train_set=x_train_lgb_dmatrix, num_boost_round=num_boost_round,)

            elif self.algorithm == 'lstm':
                # reshape input to be [samples, time steps, features]
                x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
                x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

                # create and fit the LSTM network
                clf = Sequential()
                clf.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
                clf.add(Dropout(0.4))
                clf.add(LSTM(128))
                clf.add(Dense(64, activation='relu'))
                clf.add(Dropout(0.2))
                clf.add(Dense(2, activation='softmax'))
                clf.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                clf.fit(x_train, y_train, epochs=50, batch_size=1000, verbose=2)

            # prediction
            if self.algorithm == 'lgb':
                x_test_lgb = x_test
                y_pred = clf.predict(x_test_lgb)
            elif self.algorithm == 'xgb':
                x_test_xgb = x_test
                x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)
                y_pred = clf.predict(x_test_xgb_dmatrix)
            elif self.algorithm == 'dnn':
                y_pred = dnn_model.predict(x_test)[:, 0]
            elif self.algorithm == 'rf_rf':
                x_test_rf1 = x_test[:,:self.feature1_length]
                x_test_rf2 = x_test[:,self.feature1_length:]

                y_pred_rf1 = rf1.predict_proba(x_test_rf1)[:, 1]
                y_pred_rf2 = rf2.predict_proba(x_test_rf2)[:, 1]

                y_pred = y_pred_rf1 * 0.5 + y_pred_rf2 * 0.5
            elif self.algorithm == 'xgb_xgb':
                x_test_xgb1 = x_test[:,:self.feature1_length]
                x_test_xgb1_dmatrix = xgb.DMatrix(x_test_xgb1, label=y_test)

                x_test_xgb2 = x_test[:,self.feature1_length:]
                x_test_xgb2_dmatrix = xgb.DMatrix(x_test_xgb2, label=y_test)

                y_pred_xgb1 = xgb1.predict(x_test_xgb1_dmatrix)
                y_pred_xgb2 = xgb2.predict(x_test_xgb2_dmatrix)

                # assign weight
                y_pred = y_pred_xgb1 * 0.5 + y_pred_xgb2 * 0.5

                learned_correctness, engineered_correctness, combine_correctness, learned_incorrectness, engineered_incorrectness, combine_incorrectness = self.predict_result(y_pred_lr_single=y_pred_xgb1, y_pred_xgb_single=y_pred_xgb2 ,y_pred_combine=y_pred, y_test=y_test, patch_name_test=patch_name_test)

            elif self.algorithm == 'lr_combine':
                x_test_lr = x_test[:, :self.feature1_length]

                x_test_xgb2 = x_test[:, self.feature1_length:]
                x_test_xgb2_dmatrix = xgb.DMatrix(x_test_xgb2, label=y_test)

                y_pred_lr = lr.predict_proba(x_test_lr)[:, 1]
                y_pred_xgb = xgb2.predict(x_test_xgb2_dmatrix)

                y_pred_xgb_all = lr_all.predict_proba(x_test)[:, 1]

                learned_correctness, engineered_correctness, combine_correctness, learned_incorrectness, engineered_incorrectness, combine_incorrectness = self.predict_result(y_pred_lr_single=y_pred_lr, y_pred_xgb_single=y_pred_xgb ,y_pred_combine=y_pred_xgb_all, y_test=y_test, patch_name_test=patch_name_test)

                y_pred = y_pred_xgb_all
            elif self.algorithm == 'xgb_combine':
                x_test_xgb_all_dmatrix = xgb.DMatrix(x_test, label=y_test)

                x_test_xgb1 = x_test[:, :self.feature1_length]
                x_test_xgb1_dmatrix = xgb.DMatrix(x_test_xgb1, label=y_test)

                x_test_xgb2 = x_test[:, self.feature1_length:]
                x_test_xgb2_dmatrix = xgb.DMatrix(x_test_xgb2, label=y_test)

                y_pred_xgb1 = xgb1.predict(x_test_xgb1_dmatrix)
                y_pred_xgb2 = xgb2.predict(x_test_xgb2_dmatrix)
                y_pred_xgb_all = xgb_all.predict(x_test_xgb_all_dmatrix)

                learned_correctness, engineered_correctness, combine_correctness, learned_incorrectness, engineered_incorrectness, combine_incorrectness = self.predict_result(y_pred_lr_single=y_pred_xgb1, y_pred_xgb_single=y_pred_xgb2 ,y_pred_combine=y_pred_xgb_all, y_test=y_test, patch_name_test=patch_name_test)

                y_pred = y_pred_xgb_all
            elif self.algorithm == 'rf_combine':
                x_test_rf1 = x_test[:, :self.feature1_length]
                x_test_rf2 = x_test[:, self.feature1_length:]

                y_pred_rf1 = rf1.predict(x_test_rf1)
                y_pred_rf2 = rf2.predict(x_test_rf2)
                y_pred_rf_all = rf_all.predict(x_test)

                learned_correctness, engineered_correctness, combine_correctness, learned_incorrectness, engineered_incorrectness, combine_incorrectness = self.predict_result(y_pred_lr_single=y_pred_rf1, y_pred_xgb_single=y_pred_rf2 ,y_pred_combine=y_pred_rf_all, y_test=y_test, patch_name_test=patch_name_test)

                y_pred = y_pred_rf_all
            elif self.algorithm == 'wide_deep' or self.algorithm == 'dnn_dnn' or self.algorithm == 'dnn_cnn':
                x_test_lr = x_test[:,:self.feature1_length]
                x_test_xgb = x_test[:,self.feature1_length:]
                y_pred = combine_deep_model.predict([x_test_lr, x_test_xgb])[:, 0]

            elif self.algorithm == 'dnn_dnn_venn':
                x_test_lr = x_test[:,:self.feature1_length]
                x_test_xgb = x_test[:,self.feature1_length:]

                y_pred_lr = dnn_model1.predict(x_test_lr)[:, 0]
                y_pred_xgb = dnn_model2.predict(x_test_xgb)[:, 0]
                y_pred = combine_deep_model.predict([x_test_lr, x_test_xgb])[:, 0]

                learned_correctness, engineered_correctness, combine_correctness, learned_incorrectness, engineered_incorrectness, combine_incorrectness = self.predict_result(y_pred_lr_single=y_pred_lr, y_pred_xgb_single=y_pred_xgb ,y_pred_combine=y_pred, y_test=y_test, patch_name_test=patch_name_test)

            elif self.algorithm == 'lr_xgb':
                x_test_lr = x_test[:,:self.feature1_length]
                x_test_xgb = x_test[:,self.feature1_length:]
                x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)

                y_pred_lr = lr.predict_proba(x_test_lr)[:, 1]
                y_pred_xgb = xgb_model.predict(x_test_xgb_dmatrix)

                # assign weight
                y_pred = y_pred_lr * 0.5 + y_pred_xgb * 0.5

                # venn diagram
                # diff_correct, diff_incorrect, p = self.venn_2(y_pred_lr_single=y_pred_lr, y_pred_combine=y_pred, y_test=y_test, patch_name_test=patch_name_test)

                # venn 3
                # self.venn_3(y_pred_lr_single=y_pred_lr, y_pred_xgb_single=y_pred_xgb, y_pred_combine=y_pred, y_test=y_test, patch_name_test=patch_name_test)

                learned_correctness, engineered_correctness, combine_correctness, learned_incorrectness, engineered_incorrectness, combine_incorrectness = self.predict_result(y_pred_lr_single=y_pred_lr, y_pred_xgb_single=y_pred_xgb ,y_pred_combine=y_pred, y_test=y_test, patch_name_test=patch_name_test)

            else:
                y_pred = clf.predict_proba(x_test)[:, 1]

            # auc curve
            # self.draw_auc(y_pred, y_test)

            # confusion matrix
            # self.confusion_matrix(y_pred, y_test)

            # main metrics
            auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=list(y_test), y_pred_prob=list(y_pred))

            accs.append(acc)
            prcs.append(prc)
            rcs.append(rc)
            f1s.append(f1)

            aucs.append(auc_)
            rcs_p.append(recall_p)
            rcs_n.append(recall_n)

            if self.algorithm == 'lr_xgb' or self.algorithm == 'xgb_xgb' or self.algorithm == 'dnn_dnn_venn' or self.algorithm == 'xgb_combine'  or self.algorithm == 'lr_combine':
                learned_correctness_all += learned_correctness
                engineered_correctness_all += engineered_correctness
                combine_correctness_all += combine_correctness

                learned_incorrectness_all += learned_incorrectness
                engineered_incorrectness_all += engineered_incorrectness
                combine_incorrectness_all += combine_incorrectness

            # calculate SHAP value
            # if self.algorithm == 'xgb_combine':
            #     self.shap_value(xgb1, x_test_xgb1, xgb2, x_test_xgb2, xgb_all, x_test, learned_correctness,
            #                     engineered_correctness, combine_correctness, patch_name_test)

            # elif self.algorithm == 'rf_combine':
            #     self.shap_value(rf1, x_test_rf1, rf2, x_test_rf2, rf_all, x_test, learned_correctness,
            #                     engineered_correctness, combine_correctness, patch_name_test)

        print('')
        print('{}-fold cross validation mean: '.format(self.kfold))
        print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(np.array(accs).mean()*100, np.array(prcs).mean()*100, np.array(rcs).mean()*100, np.array(f1s).mean()*100, np.array(aucs).mean()))
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean()))

        output2 = '------------------------------------\n'
        output2 += '{}-fold cross validation\n'.format(self.kfold)
        output2 += 'AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}\n\n'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean())

        if self.algorithm == 'xgb_xgb' or self.algorithm == 'lr_xgb' or self.algorithm == 'dnn_dnn_venn' or self.algorithm == 'xgb_combine' or self.algorithm == 'lr_combine':
            self.draw_venn2(learned_correctness_all, combine_correctness_all, learned_incorrectness_all, combine_incorrectness_all)
            self.draw_venn3(learned_correctness_all, engineered_correctness_all, combine_correctness_all, learned_incorrectness_all, engineered_incorrectness_all, combine_incorrectness_all)

            # # correct included
            independent_learned, independent_engineered, independent_combine = self.independent_case(learned_correctness_all, engineered_correctness_all, combine_correctness_all)
            # print('Correct Included: ')
            # print('independent_learned: {}'.format(independent_learned))
            # print('independent_engineered: {}'.format(independent_engineered))
            # print('independent_combine: {}'.format(independent_combine))
            # # incorrect excluded
            # independent_learned, independent_engineered, independent_combine = self.independent_case(learned_incorrectness_all, engineered_incorrectness_all, combine_incorrectness_all)
            # print('Correct Included: ')
            # print('independent_learned: {}'.format(independent_learned))
            # print('independent_engineered: {}'.format(independent_engineered))
            # print('independent_combine: {}'.format(independent_combine))

        return output2

    def run_slice(self, ):

        scaler = StandardScaler()
        scaler.fit_transform(self.dataset)

        # skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()

        correct_index = np.where(self.labels == 1)[0][:20]
        incorrect_index = np.where(self.labels == 0)[0]


        for i in range(self.kfold):
            incorrect_index_tmp = np.array(random.sample(list(incorrect_index), correct_index.shape[0]))
            new_index = np.concatenate((correct_index, incorrect_index_tmp),axis=0)
            new_data = self.dataset[new_index]
            new_label = self.labels[new_index]

            x_train, x_test, y_train, y_test = train_test_split(new_data, new_label, test_size=0.2)

            # x_train, y_train = self.dataset[train_index], self.labels[train_index]
            # x_test, y_test = self.dataset[test_index], self.labels[test_index]

            clf = None
            if self.algorithm == 'lr':
                clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train, y=y_train)
            elif self.algorithm == 'rf':
                clf = RandomForestClassifier(class_weight={1: 1}, n_estimators=1000).fit(X=x_train, y=y_train)
            elif self.algorithm == 'lr_rf':
                x_train_lr = x_train[:,:self.feature1_length]
                x_train_rf = x_train[:,self.feature1_length:]

                lr = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train_lr, y=y_train)
                rf = RandomForestClassifier(class_weight={1: 1}, n_estimators=1000).fit(X=x_train_rf, y=y_train)


            # prediction
            if self.algorithm == 'lr_rf':
                x_test_lr = x_test[:,:self.feature1_length]
                x_test_rf = x_test[:,self.feature1_length:]

                y_pred_lr = lr.predict_proba(x_test_lr)[:, 1]
                y_pred_rf = rf.predict_proba(x_test_rf)[:, 1]

                # assign weight
                y_pred = y_pred_lr * 0.6 + y_pred_rf * 0.4
            else:
                y_pred = clf.predict_proba(x_test)[:, 1]
            auc_, recall_p, recall_n = self.evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)

            # accs.append(acc)
            # prcs.append(prc)
            # rcs.append(rc)
            # f1s.append(f1)

            print(clf.coef_)
            aucs.append(auc_)
            rcs_p.append(recall_p)
            rcs_n.append(recall_n)

        # print('------------------------------------------------------------------------')
        # print('{}-fold cross validation'.format(self.kfold))
        # print('Accuracy: {} -- Precision: {} -- +Recall: {} -- F1: {} -- AUC: {}'.format(np.array(accs).mean(), np.array(prcs).mean(), np.array(rcs).mean(), np.array(f1s).mean(), np.array(aucs).mean()))
        print('')
        print('{}-fold cross validation mean: '.format(self.kfold))

        output2 = '------------------------------------\n'
        output2 += '{}-time slice validation\n'.format(self.kfold)
        output2 += 'AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean())

        return output2