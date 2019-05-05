import xlrd
import numpy as np
import math
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
    table = xlsx_data.sheet_by_index(1)             # 根据sheet索引获取第一个sheet

    # preocess_table_cell = lambda x : x.value()
    batch_size_arr = table.col_values(0)[1:]
    gpu_use_arr = table.col_values(1)[1:]
    gpu_memory_arr = table.col_values(2)[1:]
    # gpu_all_step_arr = table.col_values(3)[1:]
    # step_time_arr = table.col_values(8)[1:]
    return np.array(batch_size_arr), np.array(gpu_use_arr), np.array(gpu_memory_arr)

def fetch_parallel_data():
    xlsx_data = xlrd.open_workbook(data_file_name)  # 打开xlsx文件
    table = xlsx_data.sheet_by_index(3)             # 根据sheet索引获取第一个sheet

    # preocess_table_cell = lambda x : x.value()
    gpu_use_arr = np.array(table.col_values(0)[1:])
    single_time_arr = np.array(table.col_values(1)[1:])
    parallel_time_arr = np.array(table.col_values(2)[1:])
    friend_gpu_use_arr = np.array(table.col_values(3)[1:])
    friend_single_time_arr = np.array(table.col_values(4)[1:])

    return gpu_use_arr, single_time_arr, parallel_time_arr, friend_gpu_use_arr, friend_single_time_arr

def fetch_resnet_gpu_data(type):
    '''
        获取GPU使用的数据
    '''
    file_name = './gpu_{0}_info.xlsx'.format(type)
    xlsx_data = xlrd.open_workbook(file_name)
    gpu_table = xlsx_data.sheet_by_index(0)             # 根据sheet索引获取第一个sheet

    batch_size = gpu_table.col_values(0)
    gpu_used = gpu_table.col_values(1)
    gpu_memory = gpu_table.col_values(2)

    batch_size_arr = []
    gpu_used_arr = []
    gpu_memory_arr = []

    temp_batch_size = -1
    temp_gpu_used = 0
    temp_gpu_used_count = 0
    for _ in range(len(batch_size)):
        if(temp_batch_size != batch_size[_] and temp_batch_size != -1):
            batch_size_arr.append(temp_batch_size)
            gpu_used_arr.append(temp_gpu_used / temp_gpu_used_count)
            gpu_memory_arr.append(gpu_memory[_ - 1])
            temp_gpu_used = 0
            temp_gpu_used_count = 0
        else:
            if(gpu_used[_] > 0.1):
                temp_gpu_used += gpu_used[_]
                temp_gpu_used_count += 1

        temp_batch_size = batch_size[_]

    batch_size_arr.append(batch_size[-1])
    gpu_used_arr.append(temp_gpu_used / temp_gpu_used_count)
    gpu_memory_arr.append(gpu_memory[-1])

    return np.array(batch_size_arr), np.array(gpu_used_arr), np.array(gpu_memory_arr)


def fetch_resnet_steps_data(type):
    '''
        获取步数和时间的数据
    '''
    file_name = './gpu_{0}_info.xlsx'.format(type)
    xlsx_data = xlrd.open_workbook(file_name)
    
    steps_table = xlsx_data.sheet_by_index(1)
    batch_size = steps_table.col_values(0)
    steps = steps_table.col_values(1)
    all_times = steps_table.col_values(2)

    s_batch_size_arr = []
    s_steps_arr = []
    s_times_arr = []
    inc_step = 1
    if(len(batch_size) % inc_step + len(steps) % inc_step + len(steps) % inc_step > 0):
        print("Something Error")
    else:
        for _ in range(len(batch_size) // inc_step):
            s_batch_size_arr.append(batch_size[_ * inc_step])
            s_steps_arr.append(steps[_ * inc_step])
            s_times_arr.append(all_times[_ * inc_step])
    return np.array(s_batch_size_arr), np.array(s_steps_arr), np.array(s_times_arr)

def curve_model(x, y, model, isShow=False, title="Y value"):
    popt, pcov = curve_fit(model, x, y)
    y_predit = None
    if(len(popt) == 1):
        y_predit = model(x, popt[0])
    elif(len(popt) == 2):
        y_predit = model(x, popt[0], popt[1])
    elif(len(popt) == 3):
        y_predit = model(x, popt[0], popt[1], popt[2])
    elif(len(popt) == 4):
        y_predit = model(x, popt[0], popt[1], popt[2], popt[3])
    elif(len(popt) == 5):
        y_predit = model(x, popt[0], popt[1], popt[2], popt[3], popt[4])
    else:
        y_predit = model(x)

    if(isShow):
        p1 = plt.plot(x, y, label='Real Data')
        p2 = plt.plot(x, y_predit, label='Curve Data')
        plt.legend([p1, p2], labels=['Real Data', 'Curve Data'])
        plt.xlabel("batch_size")
        plt.ylabel(title)
        plt.show()

    return popt

def curve_parallel_time():
    p_gpu_use_arr, p_single_time_arr, p_parallel_time_arr, friend_gpu_use_arr, friend_single_time_arr = fetch_parallel_data()
    '''
        模拟的模型为: T并行 = (w * job利用率和 + b) * T单独时间和 * 竞争job的利用率/job利用率和 + c
                             ---------------------------------    -------------------------  
                                           ☝                                 ☝
                                      计算出总时间                    最终的时间和利用率成反比
     '''
    use_sum = p_gpu_use_arr + friend_gpu_use_arr
    time_sum = p_single_time_arr + friend_single_time_arr
    use_scale = friend_gpu_use_arr / use_sum
    time_scale = p_parallel_time_arr / p_single_time_arr

    def model (x, w, b):
        return w * x + b
    
    # popt, pcov = curve_fit(model, use_scale, time_scale)
    # xaxis = [ (i + 1) for i in range(len(p_gpu_use_arr))]
    # y_predict = model(use_scale, popt[0], popt[1])
    # plt.plot(xaxis, time_scale)
    # plt.plot(xaxis, y_predict)
    plt.scatter(p_gpu_use_arr, p_parallel_time_arr/p_single_time_arr)
    plt.show()    

    print(gpu_use_arr)

def model_gpu_use(choice=1):
    '''
        拟合GPU利用率曲线
    '''
    if(choice == 1):
        return lambda x, b: x**2 + b  # 不可能是这个了
    elif(choice == 2):
        return lambda x, w, b: w * np.log2(x) + b  # 一般 不能用
    elif(choice == 3):
        return lambda x, w, b: 1 - 1 / (math.e ** (x * w + b) + 1)  # 一般般
    elif(choice == 4):
        return lambda x, w, b: 1 - 1 / (w * x + b) ** 2 + 1 
    elif(choice == 5):
        return lambda x, w, b: 1 - 1 / (math.e ** (np.log2(x) * w + b) + 1)  
    elif(choice == 6):
        return lambda x, w, b: 1 - 1 / (math.e ** (-(np.log2(x) * w + b) + 1))   

def model_gpu_memory(choice=1):
    '''
        显存占用和batch_size 之间的模型
    '''
    if(choice == 1):
        return lambda x, w, b: w * x + b
    elif(choice == 2):
        # 效果还行
        return lambda x, w, b: w * np.floor(np.log2(x)) + b ** np.floor(np.log2(x))
    elif(choice == 3):
        return lambda x, w, b: w * np.floor(np.log2(x)) + b * np.floor(np.log2(x))  # 效果不行 有负值


def model_step_time(choice):
    '''
        拟合单步时间曲线
    '''
    if(choice == 1):
            return lambda x, w, b: w * x + b

def model_total_time(choice):
    '''
        拟合总时间曲线
    '''
    if(choice == 1):
            return lambda x, w, b: w * x + b


def model_all_step(choice=1):
    '''
        拟合需要多少步数
    '''
    if(choice == 1):
        return lambda x, w, b: w * x + b
    elif(choice == 2):
        return lambda x, w, w1, b: w * ( x ** -2) + w1 / x + b # 效果韩星
    elif(choice == 3):
        return lambda x, w, b: w * ( x ** -2) + b  # 效果不行 有负值
    elif(choice == 4):
        return lambda x, w, b: w / x + b
    elif(choice == 5):
        return lambda x, w, w1, b: w /(x ** 2) + w1 / x + b        

def model_parallel_time(choice=1):
    '''
        拟合并行运行时 单独时长 和 并行时长之间的关系
        拟合曲线 y 表示 并行时长/单独时长
        拟合曲线 x 表示 job对应的gpu利用率
    '''
    per_gpu = 0.5
    if(choice == 1):
        return lambda x, w, b: w * x / per_gpu + b
    elif(choice == 2):
        return lambda x, w, b: w * x /per_gpu + b 

if __name__ == "__main__":
    batch_size_arr, gpu_used_arr, gpu_memory_arr = fetch_resnet_gpu_data(47)
    s_batch_size_arr, steps_arr, time_arr = fetch_resnet_steps_data(47)

    model = model_gpu_memory(2)
    parm = curve_model(batch_size_arr, gpu_memory_arr, model, isShow=False, title= "GPU Memroy")
    

    # model = model_all_step(2)
    # parm = curve_model(s_batch_size_arr, steps_arr, model, isShow=True)

    model = model_step_time(1)
    parm = curve_model(s_batch_size_arr, time_arr / steps_arr, model, isShow=False, title="Step Time")

    model = model_total_time(1)
    parm = curve_model(s_batch_size_arr, time_arr, model, isShow=True, title="Total Time")

    model = model_gpu_use(5)
    parm = curve_model(batch_size_arr, gpu_used_arr, model, isShow=True, title= "GPU Used")


    # s_batch_size_arr, steps_arr, time_arr = fetch_resnet_steps_data(47)
    #model = model_gpu_memory(2)
    #parm = curve_model(batch_size_arr, gpu_memory_arr, model, isShow=True, title= "GPU Memroy")
    model = model_all_step(5)
    parm = curve_model(s_batch_size_arr, steps_arr, model, isShow=True, title = "All Steps")
    print(parm)


    # 获取并行运行的数据
    # curve_parallel_time()
