import os 
import re

file = open("training_data.txt", "w").close()
file = open("training_data.txt", "a")
for filename in os.listdir("wordLemPoS"):
    f = open("wordLemPoS/"+filename, "r")
    for i, line in enumerate(f):
        if i > 4:
            try:
                line_split = line.split()
                if "_" in line_split[2]:
                    line_split[2] = line_split[2][0:line_split[2].index("_")]   
                if line_split[2][-1].isdigit() and line_split[2][-2].isdigit():
                    line_split[2] = line_split[2][0:-2]
                line_split[2] = re.sub(r'[^a-zA-Z0-9]','', line_split[2])
                file.write(line_split[0]+" "+line_split[2]+"\n")
            except:
                print(line) 
    f.close()
file.close()
        
