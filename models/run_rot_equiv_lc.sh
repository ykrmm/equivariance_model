gamma=0.95
mtask=False
rotate=True
gpu=1
split=True
split_ratio=0.3

python3 rot_equiv_model_landscape.py --learning_rate 0.001 --Loss KL --gamma $gamma --multi_task $mtask --scheduler True --batch_size 4 --nw 4 --gpu $gpu --rot_cpu True --n_epochs 120 --model FCN --model_name rot_equiv_lc --pi_rotate False --rotate $rotate --angle_max 360  --split $split --split_ratio $split_ratio --landcover True --save_dir /share/homes/karmimy/equiv/save_model/rot_equiv_lc --save_all_ep False --save_best True --load_last_model False 
