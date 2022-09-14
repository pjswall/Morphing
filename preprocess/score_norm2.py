#################FLOAT 3RD PLACE DECIMAL CONVERSION#############

import csv

with open('new_norm.csv', 'r') as f_input, open('precise_3.csv', 'w') as f_output:
    csv_input = csv.reader(f_input)
    csv_output = csv.writer(f_output)
    # print(csv_input)
    # print(csv_output)
    csv_output.writerow(next(csv_input))    # write header

    for cols in csv_input:
        for i in range(3, 12):
            cols[i] = '{:.3f}'.format(float(cols[i]))
        csv_output.writerow(cols)


###################Colummn REORDERING CODE##############

# import csv

# with open('Scores_norm.csv', 'r') as infile, open('Scores_norm_aligned.csv', 'a') as outfile:
#     # output dict needs a list for new column ordering
#     fieldnames = ['','imgname' , 'isMorph' ,'gt1_gt2' ,'gt1_input'  , 'gt2_input'  , 'output1_input'  , 'output2_input'  , 'gt1_output1' ,'gt1_output2' ,'gt2_output1' ,'gt2_output2', 'output1_output2']
#     writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#     # reorder the header first
#     writer.writeheader()
#     for row in csv.DictReader(infile):
#         # writes the reordered rows to the new file
#         writer.writerow(row)