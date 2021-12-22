from tqdm import tqdm
import csv
import json

classes = {
    1: "fruit_woodiness", 2: "fruit_brownspot", 3: "fruit_healthy"
}

res = csv.writer(open('submission.csv', 'w', newline=''))
f_json = json.load(open('xxx.bbox.json', 'r'))

res.writerow(['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax'])

for r in f_json:
    res.writerow([r['image_id'], classes[r['category_id']], r['score'], r['bbox'][1], r['bbox'][0], r['bbox'][1]+r['bbox'][3], r['bbox'][0]+r['bbox'][2]])