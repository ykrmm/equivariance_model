gpu=0
rotate=False
split=True
split_ratio=0.1


python3 fully_supervised_models.py --learning_rate 10e-4 --scheduler True --batch_size 4 --nw 4 --gpu $gpu  --n_epochs 120 --model DLV3 --model_name dlv3_fully_sup --rotate $rotate --scale True --benchmark True  --split $split --split_ratio $split_ratio --save_dir /share/homes/karmimy/equiv/save_model/fully_supervised --save_best True  


