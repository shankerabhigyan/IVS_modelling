import json

p1 = 'data_new_15_31_01_24.json'
p2 = 'data_new.json'
p3 = 'data.json'

# append p1 to p2
with open(p1, 'r') as f:
    data1 = json.load(f)

with open(p2, 'r') as f:
    data2 = json.load(f)

data2.extend(data1)

with open(p3, 'w') as f:
    json.dump(data2, f)
# End of merge_json.py