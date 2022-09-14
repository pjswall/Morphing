import os
from termios import FF1
from tqdm import tqdm
import re


dir1 = "./P1_done_cropping/Cropped_Images"
dir2 = "./P2_done_cropping/Cropped_Images"

p = re.compile(r'\d+')
q = re.compile(r"d\d+")


mor1_titles = os.listdir(dir1)
mor2_titles = os.listdir(dir2)

f_n = "04203d134-vs-04315d18"

f1 = set()
f2 = set()

f3 = set()
f4 = set()

for i in tqdm(mor1_titles):
    m = re.search(p,i)
    n = re.search(q,i)

    if m is not None:
        m1 = m.group(0)
        f1.add(m1)
        
    
    if n is not None:
        n1 = n.group(0)
        # n11 = n1[1:]
        f2.add(n1)

print(f1)
print(len(f1))

print("========================================================F1-DONE-NOW-F2-==============================================================")

print(len(f2))
print(f2)


f3 =  set()
f4 =  set()


for i in tqdm(mor2_titles):
    m = re.search(p,i)
    n = re.search(q,i)

    if m is not None:
        m1 = m.group(0)
        f3.add(m1)
        
    
    if n is not None:
        n1 = n.group(0)
        # n11 = n1[1:]
        f4.add(n1)

print(len(f3))
print(f3)

print("========================================================F3-DONE-NOW-F4-==============================================================")

print(len(f4))
print(f4)



print(f1.isdisjoint(f3))

print(f2.isdisjoint(f4))