import xlrd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data_file_name = "./data.xlsx"

def data_fetch():
    '''
        为了处理获取xlsx里面的数据
        @ return bath_size_arr    对应的所有的batch_size 列表
        @ return gpu_user_arr     GPU利用率列表
        @ return gpu_memory_arr   显存利用量列表
        返回的列表长度应该保持一致
    '''
    xlsx_data = xlrd.open_workbook(data_file_name)  # 打开xlsx文件
    table = xlsx_data.sheet_by_index(0)             # 根据sheet索引获取第一个sheet

    # preocess_table_cell = lambda x : x.value()
    batch_size_arr = table.col_values(0)[1:]
    gpu_use_arr = table.col_values(1)[1:]
    gpu_memory_arr = table.col_values(3)[1:]
    return np.array(batch_size_arr), np.array(gpu_use_arr), np.array(gpu_memory_arr)

def model_memory(x, w, b, theta):
    '''
        拟合显存占用曲线
        @ param x      对应batch_size
        @ param w      对应样本大小
        @ param b      对应模型显存占用
        @ param theta  对应缩减系数,因为CUDA本身会有一些占用
    '''
    return ((w * x) + b) * theta

def model_gpu_use(x):
    '''
        拟合GPU利用率曲线
    '''
    return x**2

def model_total_time(x):
    '''
        拟合时间曲线
    '''
    return x**2


if __name__ == "__main__":
    batch_size_arr, gpu_use_arr, gpu_memory_arr = data_fetch()
    popt, pcov = curve_fit(model_memory, batch_size_arr, gpu_memory_arr)
    model_memory_w, model_memory_b, model_memory_theta= popt
    print(popt)
    memory_result = model_memory(batch_size_arr, model_memory_w, model_memory_b, model_memory_theta)
    plt.plot(batch_size_arr, gpu_memory_arr)
    plt.plot(batch_size_arr, memory_result)
    plt.show()