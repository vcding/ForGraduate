import numpy as np
import time
import sys
import tkinter as tk

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
MAX_RESOURCE = 100

SCREEN_RESOURCE = SCREEN_WIDTH / MAX_RESOURCE
UNIT = SCREEN_WIDTH // 10

JOBS = [
    {'res': 20, 'time': 10},
    {'res': 20, 'time': 8},
    {'res': 40, 'time': 5},
    {'res': 40, 'time': 10},
    {'res': 20, 'time': 5},
    {'res': 70, 'time': 20},
    {'res': 24, 'time': 20},
    {'res': 65, 'time': 20},
    {'res': 54, 'time': 20},
]
COLOR = ['#ff7f50', '#87cefa', '#da70d6', '#32cd32', '#6495ed',
         '#ff69b4', '#ba55d3', '#cd5c5c', '#ffa500', '#40e0d0',
         '#1e90ff', '#ff6347', '#7b68ee', '#00fa9a', '#ffd700',
         '#6699FF', '#ff6666', '#3cb371', '#b8860b', '#30e0e0']


class SimulateGPU(tk.Tk, object):
    def __init__(self):
        super(SimulateGPU, self).__init__()
        self.title("Simulate GPU")
        self.geometry('{0}x{1}'.format(SCREEN_WIDTH , SCREEN_HEIGHT))
        self._build_gui()
        self.timelines = {"0": MAX_RESOURCE}
        self.timeRecord = [0]

    def _build_gui(self):
        self.canvas = tk.Canvas(
            self, bg='white', height=SCREEN_HEIGHT, width=SCREEN_WIDTH)

        for c in range(0, SCREEN_WIDTH // 10):
            x0, y0, x1, y1 = c * UNIT, 0, c * UNIT, SCREEN_HEIGHT
            self.canvas.create_line(x0, y0, x1, y1)
        self.canvas.pack()

    def push_job(self, job_res, job_time):
        has_push = False
        target_timeStap = 0

        for timestap in self.timeRecord:
            resource = self.timelines[str(timestap)]
            if(resource >= job_res and not has_push):
                target_timeStap = timestap + job_time
                self.timelines[str(target_timeStap)] = resource
                self.timelines[str(timestap)] = resource - job_res
                has_push = True
                print(MAX_RESOURCE - resource, timestap, job_res, job_time)
                self.rect = self.canvas.create_rectangle(
                    (MAX_RESOURCE - resource) * SCREEN_RESOURCE, timestap * SCREEN_RESOURCE,
                    (MAX_RESOURCE - resource + job_res) * SCREEN_RESOURCE, target_timeStap * SCREEN_RESOURCE,
                    fill='red')
                continue
            if(has_push and timestap > target_timeStap):
                break
            if(has_push and timestap < target_timeStap):
                self.timelines[str(timestap)] -= job_res
        if(target_timeStap >= max(self.timeRecord)):
             self.timelines[str(target_timeStap)] = MAX_RESOURCE

        if(not target_timeStap in self.timeRecord):
            self.timeRecord.append(target_timeStap)
            self.timeRecord.sort()
        print(self.timelines)
        # print(self.timeRecord)
        

    def render(self):
        time.sleep(1)
        self.update()

def update():
    for job in JOBS:
        env.push_job(job['res'], job['time'])
        env.render()

if __name__ == "__main__":
    env = SimulateGPU()
    env.after(100, update)
    env.mainloop()
