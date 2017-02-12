# wordvector-based
import os

home = os.environ["HOME"]
#process glove
glove_vector = {}
glove_path = "".join([home, "/data/glove/glove.twitter.27B.200d.txt"])
for line in open(glove_path):
    item = line.split()
    glove_vector[item[0]] = [float(val) for val in item[1:]]
    print(glove_vector[item[0]])

print(len(glove_vector))

#utility

def compareVector(vec1, vec2):
    return 0