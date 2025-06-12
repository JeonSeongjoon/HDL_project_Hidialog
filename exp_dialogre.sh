export Albert_LARGE_DIR=/teamspace/studios/this_studio/HiDialog/pre-trained_model/Albert_v2
export Lightning_Hidialog_DIR=/teamspace/studios/this_studio/HiDialog

nohup python $Lightning_Hidialog_DIR/run_classifier.py --do_train --do_eval --encoder_type Albert  --data_dir $Lightning_Hidialog_DIR/datasets/DialogRE --data_name DialogRE   --vocab_file $Albert_LARGE_DIR/spiece.model  --config_file $Albert_LARGE_DIR/config.json   --init_checkpoint $Albert_LARGE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 8   --learning_rate 1e-5   --num_train_epochs 5  --warmup_proportion 0.2 --output_dir DialogRE --gradient_accumulation_steps 4 --gpu 0 > $Lightning_Hidialog_DIR/logs/DialogRE.log 2>&1 &


#### revision list ####
# --max_seq_length 512   
# --train_batch_size 12
# --learning_rate 7.5e-6 

#### eliminations ####
# --merges_file $Albert_LARGE_DIR/merges.txt