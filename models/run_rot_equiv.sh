python3 -m cProfile -s tottime rot_equiv_model.py --learning_rate 10e-4 --Loss KL --gamma 0.5 --multi_task True --rot_cpu True --batch_size 4 --nw 4 --gpu 0  --n_epochs 5 --model DLV3 --model_name equiv_dlv3 --rotate False --benchmark True --split False --extra_coco False --save_dir /share/homes/karmimy/equiv/save_model/rot_equiv --save_all_ep False --save_best True --load_last_model False