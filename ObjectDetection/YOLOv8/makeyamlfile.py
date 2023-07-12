import yaml

data = {
    "train" : '/mnt/tram_dataset/source/bongo_tld/tld_sample/train/',
        "val" : '/mnt/tram_dataset/source/bongo_tld/tld_sample/valid/',
        "test" : '/mnt/tram_dataset/source/bongo_tld/tld_sample/test/', 
        "names" : {0 : 'red', 1 : 'green'}}


with open('./bongo_tld.yaml', 'w') as f :
    yaml.dump(data, f)

with open('./bongo_tld.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)