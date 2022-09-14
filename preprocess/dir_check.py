# from pathlib import Path
import glob

dir_names = []

for f in glob.glob('./**/*.JPG',recursive=True):
    dir_names.append(f[-12:-4])
    # dir_names.append('\\n')

print(len(dir_names))
print(dir_names)


with open('file_name.txt', 'w') as f:
    for i in range(len(dir_names)):
        f.write(dir_names[i])
        f.write('\n')

# for file in glob.glob("./Fall2002/**/*.jpg",recursive=True):
    # print(file)
