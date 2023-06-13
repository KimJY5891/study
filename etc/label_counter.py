import os
import glob 
from collections import Counter

dir_path = '경로'
file_name = glob.glob(os.path.join(dir_path,'*.jpg'))

labels  = [ ]
for file_name in file_name :
    name_withoust_extension = os.path.splitext(os.path.basename(file_name))[0]
    parts = name_withoust_extension.split('_')
    label = parts[0]
    labels.append(label)

label_counts = Counter(labels)
print('label의 사진 갯수 : ',label_counts)
