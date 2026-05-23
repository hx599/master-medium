# -*- coding: utf-8 -*-
"""把每次实验的指标、时间和附加统计结果写入文本日志。"""
import numpy as np


def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, cm, training_time_ae, testing_time_ae, path):
    """把一组实验的主指标、时间和混淆矩阵统计写入日志文件。"""
    # 以追加模式写文件，这样多次实验结果会接在同一个日志里。
    f = open(path, 'a')

    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    sentence6 = 'Total average Training time is: ' + str(np.mean(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.mean(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)

    # 统计各类别精度的均值和方差。
    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements accuracy: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements accuracy: " + str(element_std) + '\n'
    f.write(sentence9)

    f.write("Mean of confusion matrix: " +'\n') 
    # 多次实验的混淆矩阵也取一个平均，方便总体观察。
    cm = np.array(cm)
    mean_cm = np.mean(cm, axis = 0)
    for i in range(mean_cm.shape[0]):
        f.write(str(mean_cm[i]) + '\n')
    f.write("########################################################################################################" +'\n'+ '\n') 
    f.close()


def record_add(add_content, path):
    """把 seen / unseen / harmonic 等附加指标写入日志文件。"""
    # 这里主要用于记录 seen / unseen / harmonic 这类附加指标。
    f = open(path, 'a')    
    element_mean = np.mean(add_content, axis=0)
    element_std = np.std(add_content, axis=0)
    sentence1 = "Mean of all GZSL accuracies: " + str(element_mean) + '\n'
    f.write(sentence1)
    sentence2 = "Standard deviation of all GZSL accuracies: " + str(element_std) + '\n'
    f.write(sentence2)
    f.write("########################################################################################################" +'\n'+ '\n') 
    f.close()
    
    



