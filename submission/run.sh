# Training
python tools/train.py conf.py --no-validate

# Inference
python tools/test.py conf.py weight.pth --format-only --eval-options "jsonfile_prefix=xxx"

# Get result
python submit_fruit.py
