import subprocess
import time
import os

def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False


processes = list()
N = 2#how many process running at the same time
queue = list()

folder = "oneVideo35"

#generate command
for video in sorted(os.listdir(folder)):
    queue.append(['python', 'test_sortOH.py', '--video', os.path.join(folder, video)])

#log file
file = open("testResult_sortOH.log", 'w')

subprocess.run('rm videoTest_sortOH/*', shell=True)#remove old videos



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

file.close()