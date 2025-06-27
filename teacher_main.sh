# TODO: setting1
CUDA_VISIBLE_DEVICES=0 python teacher_main.py --dir_data='./datasets' \
               --n_GPUs=1 \
               --model='teacher' \
               --scale='4' \
               --blur_type='iso_gaussian' \
               --noise=0.0 \
               --sig_min=0.2 \
               --sig_max=4.0

# TODO: setting2
#CUDA_VISIBLE_DEVICES=0 python teacher_main.py --dir_data='./datasets' \
#               --n_GPUs=1 \
#               --model='teacher' \
#               --scale='4' \
#               --blur_type='aniso_gaussian' \
#               --noise=25.0 \
#               --lambda_min=0.2 \
#               --lambda_max=4.0
