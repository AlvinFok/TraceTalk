import subprocess
import time
import os

def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False




processes = list()
N = 5#how many process running at the same time
queue = list()

folder = "usedVideos/"

#generate command
for video in os.listdir(folder):
    queue.append(['python', 'test_deepsort.py', '--video', os.path.join(folder, video)])

#log file
file = open("testResult_deepsort.log", 'w')

subprocess.run('rm videoTest_deepsort/*', shell=True)#remove old videos


for process in queue:
    p = subprocess.Popen(process, stdout=file)
    processes.append(p)
    if len(processes) == N:
        wait = True
        while wait:
            done, num = check_for_done(processes)

            if done:
                processes.pop(num)
                wait = False
            else:
                time.sleep(60) # set this so the CPU does not go crazy


if len(processes) != 0:#wait last processes
    wait = True
    while wait:
        done, num = check_for_done(processes)

        if done:
            processes.pop(num)
            wait = False
        else:
            time.sleep(60) # set this so the CPU does not go crazy


file.close()