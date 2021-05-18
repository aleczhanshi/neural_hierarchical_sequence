#!/bin/bash

benchmark=soplex_66B # specify the directory of input data

keep_ratio=0.8  # dropout ratio
step_size=16  # sequence length

# predict next address by this PC, or global
# 0: predict global stream
# 1: predict PC localized stream
pc_localization=0

page_embed_size=128  # page embedding size

# offset embedding size, 100 times of page for conditional attention embedding
multiple=100 
offset_embed_size=$((${page_embed_size}*${multiple})) 
offset_embed_size_internal=${page_embed_size}  # embedding size to perform attention

# other hyper parameters
pc_embed_size=64 # pc embedding size
learning_rate_decay=2 # learning rate decay ratio
lstm_layer=1  # number of lstm layers
batch_size=512  # batch size
use_pc_history=1  # use pc sequence 

# deprecated
complete_embedding=0  # not used
complete_loss=0  # not used
trace_length=100 # not used
# output results to a directory, may skip this
# folder=results/${benchmark}_${trace_length}
# model_no=`bash next_model.sh ${folder}`
# model_dir="${folder}/${model_no}"
# mkdir -p ${model_dir}

# nohup python3 -u nhs.py --trace_length ${trace_length} --benchmark ${benchmark} --page_embed_size ${page_embed_size} --pc_embed_size ${pc_embed_size} --offset_embed_size ${offset_embed_size} --offset_embed_size_internal ${offset_embed_size_internal} --lstm_size 256 --keep_ratio ${keep_ratio} --step_size ${step_size} --learning_rate 0.001 --learning_rate_decay ${learning_rate_decay} --complete_loss ${complete_loss} --complete_embedding ${complete_embedding} --batch_size ${batch_size} --use_pc_history ${use_pc_history} --lstm_layer ${lstm_layer} --pc_localization ${pc_localization} > ${model_dir}/${benchmark}.txt 2>&1 &
nohup python3 -u nhs.py --trace_length ${trace_length} --benchmark ${benchmark} --page_embed_size ${page_embed_size} --pc_embed_size ${pc_embed_size} --offset_embed_size ${offset_embed_size} --offset_embed_size_internal ${offset_embed_size_internal} --lstm_size 256 --keep_ratio ${keep_ratio} --step_size ${step_size} --learning_rate 0.001 --learning_rate_decay ${learning_rate_decay} --complete_loss ${complete_loss} --complete_embedding ${complete_embedding} --batch_size ${batch_size} --use_pc_history ${use_pc_history} --lstm_layer ${lstm_layer} --pc_localization ${pc_localization} 2>&1 &
