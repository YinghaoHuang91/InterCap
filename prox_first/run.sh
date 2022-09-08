ID=0
#IN_DIR=#PATH_TO_DATA
IN_DIR=../Data/recordings/subject01_motion01_seg01
#OUT_DIR=#OUTPUT_PATH
OUT_DIR=../Output

python prox/main.py --batch_id $ID --config cfg_files/SMPLifyD.yaml --recording_dir $IN_DIR  --output_folder $OUT_DIR --visualize=False --vposer_ckpt ../models/vposer_v1_0/ --part_segm_fn ../models/smplx_parts_segm.pkl --model_folder ../models
