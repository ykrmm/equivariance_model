#parameters
save_repo=/share/homes/karmimy/equiv/save_model/fully_supervised_lc
name=fcn_fully_sup_lc
rotate=False
gpu=3
split=True
split_ratio=0.1

#running command
python3  fully_supervised_models_lc.py --auto_lr False --learning_rate 0.001 --scheduler True --batch_size 4 --nw 4 --gpu $gpu --n_epochs 50 --model FCN --model_name $name --landcover True --rotate $rotate --pi_rotate False --p_rotate 0.25 --scale False --size_crop 480 --benchmark True --split $split --split_ratio $split_ratio --save_dir $save_repo --save_best True 
