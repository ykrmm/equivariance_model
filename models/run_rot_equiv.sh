gamma=0.5
mtask=False
rotate=True
gpu=0
split=True
split_ratio=0.3
Loss=CE




python3 rot_equiv_model.py --learning_rate 0.001 --Loss $Loss --gamma $gamma --multi_task $mtask --rot_cpu True --batch_size 4 --nw 5 --gpu $gpu  --n_epochs 130 --model DLV3 --model_name equiv_dlv3 --rotate $rotate --benchmark True --split $split --split_ratio $split_ratio --extra_coco False --save_dir /share/homes/karmimy/equiv/save_model/rot_equiv --save_all_ep False --save_best True --load_last_model False
