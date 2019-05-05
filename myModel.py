import numpy as np
import math
import matplotlib.pyplot as plt

def gpu_memory(b):
    x = np.floor(np.log2(b))
    return 286.83 * x + 2.58 ** x

def gpu_used(b):
    return 1 - 1 / (math.e ** (1.1 * np.log2(b) - 3.06) + 1)

def step_time(b):
    return (0.87 * b + 7.02) / 1000

def train_steps(b):
    return (-98300 / ( b * b) + 30300 / b - 4.34) * 100

def draw_lines():
    b = np.array([ i * 2 for i in range(8, 217)])
    memory = gpu_memory(b)
    used = gpu_used(b)
    stept = step_time(b)
    steps = train_steps(b)

    plt.subplot(2,2,1)
    plt.plot(b, memory)
    
    plt.subplot(2,2,2)
    plt.plot(b, used)

    plt.subplot(2,2,3)
    plt.plot(b, stept)

    plt.subplot(2,2,4)
    plt.plot(b, steps)
    plt.show()

def find_the_batch_size():
    b = np.array([ i * 2 for i in range(8, 217)])
    memory = gpu_memory(b) / 8000 
    used = gpu_used(b)

    stept = step_time(b)
    steps = train_steps(b)
    total_time = stept * steps;

    total_time = total_time / (max(total_time) * 1.1)

    resource = memory * used * total_time

    print(min(resource), max(resource))

    plt.subplot(2,2,1)
    plt.plot(b, memory)
    plt.title("GPU_MEMORY")

    plt.subplot(2,2,2)
    plt.plot(b, used)
    plt.title("GPU_USED")

    plt.subplot(2,2,3)
    plt.plot(b, total_time)
    plt.title("TOTAL_TIME")

    plt.subplot(2,2,4)
    plt.plot(b, resource)
    plt.title("RESOURCE")
    plt.show()

if __name__ == "__main__":
    # draw_lines()
    find_the_batch_size()