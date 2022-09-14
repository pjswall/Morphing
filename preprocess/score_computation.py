import os
import re
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

pattern = re.compile(r'\d+\_\d+\_q?\d+')

folders = ['Mor_morph', 'Mor_nonmorph']

data = {
	'imgname' : [],
	'isMorph' : [],
	'gt1_gt2' : [],
	'gt1_input' : [],
	'gt2_input' : [],
	'output1_input' : [],
	'output2_input' : [],
	'gt1_output1' : [],
	'gt1_output2' : [],
	'gt2_output1' : [],
	'gt2_output2' : [],
	'output1_output2' : []
}

def calculate_score(path1,path2):
    result = DeepFace.verify(path1,path2,model_name = 'ArcFace',enforce_detection=False)
    return result['distance']


f1=set()

for folder in tqdm(folders):
	is_morph_check = folder.replace('Mor_', '')
	for filename in tqdm(os.listdir(folder)):
		m = re.search(pattern, filename)
		if m is not None:
			match_str = m.group(0)
			# print(match_str)
			match_str = match_str.strip()	
			# print(match_str)
			f1.add(match_str)

			if(match_str not in data['imgname']):
				## column1
				data['imgname'].append(match_str)

				## column2
				if(is_morph_check=="morph"):
					data['isMorph'].append(1)
				else:
					data['isMorph'].append(0)


				### Image files for processing
				gt1 = folder+"/"+match_str+"_gt1.png"
				gt2 = folder+"/"+match_str+"_gt2.png"
				Input = folder+"/"+match_str+"_input.png"
				output1 = folder+"/"+match_str+"_output1.png"
				output2 = folder+"/"+match_str+"_output2.png"


				data['gt1_gt2'].append('{0:.3g}'.format(calculate_score(gt1,gt2)))
				data['gt1_input'].append('{0:.3g}'.format(calculate_score(gt1,Input)))
				data['gt2_input'].append('{0:.3g}'.format(calculate_score(gt2,Input)))
				data['output1_input'].append('{0:.3g}'.format(calculate_score(output1,Input)))
				data['output2_input'].append('{0:.3g}'.format(calculate_score(output2,Input)))
				data['gt1_output1'].append('{0:.3g}'.format(calculate_score(gt1,output1)))
				data['gt1_output2'].append('{0:.3g}'.format(calculate_score(gt1,output2)))
				data['gt2_output1'].append('{0:.3g}'.format(calculate_score(gt2,output1)))
				data['gt2_output2'].append('{0:.3g}'.format(calculate_score(gt2,output2)))
				data['output1_output2'].append('{0:.3g}'.format(calculate_score(output1,output2)))

df = pd.DataFrame(data = data, columns = list(data.keys()))
print(df.head())
print(df.describe())
df.to_csv('scores.csv')
