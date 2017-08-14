import os 

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
                file.write(line_split[0]+" "+line_split[2]+"\n")
            except:
                print(line) 
    f.close()
file.close()
        
