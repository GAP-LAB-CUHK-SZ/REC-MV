DEVICES=${1}
SUBJ=${2}

bash ./scripts/parsing_mask.sh ${DEVICES} ./configs/people_snapshot/${SUBJ}.conf ./people_snapshot_public_proprecess/${SUBJ}/
python ./tools/parsing_mask_to_fl.py --parsing_type ATR  --input_path ./people_snapshot_public_proprecess/${SUBJ}/ --output_path ./people_snapshot_public_proprecess/${SUBJ}/mask2fl



