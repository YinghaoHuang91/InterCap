ID=0
IN_FOLDER=../Data/recordings/subject01_motion01_seg01
OUT_FOLDER=../Output
MODEL_FOLDER=../models

python prox/main.py --batch_id 0 --config cfg_files/SMPLifyD.yaml --recording_dir $IN_FOLDER  --output_folder $OUT_FOLDER --visualize=False --vposer_ckpt $MODEL_FOLDER/vposer_v1_0/ --part_segm_fn $MODEL_FOLDER/smplx_parts_segm.pkl --model_folder $MODEL_FOLDER --read_depth True --read_mask True --use_vposer True --contact True
