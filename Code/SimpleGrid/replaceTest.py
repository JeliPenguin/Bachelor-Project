import sys
import time

for i in range(5):
    content = "Im at" + str(i)
    sys.stdout.write("\r%s"%content)
    sys.stdout.flush()
    time.sleep(2)