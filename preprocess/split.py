import os 
import re


with open("file_name.txt",'r') as f:
    lines =f.readlines()
    # print(lines[:100])

p = re.compile(r'\d+')
q = re.compile(r"d\d+")

f1 =set()
f2 = set()

for i in lines:
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
print(f2)