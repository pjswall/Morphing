import os 
import re

dir_name =  "sftp://prateekj@ada/home2/prateekj/frgc2/all_imgs"

p = re.compile(r'\d+')
q = re.compile(r'd\d+')


all_img_titles =  os.listdir(dir_name)


f1 = set()
f2 = set()


for i in all_img_titles:
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

