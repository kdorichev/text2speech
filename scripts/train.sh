#!/bin/bash

# Adjust env variables to maintain the global batch size
#
#    NGPU x BS x GRAD_ACC = 256.

[ ! -n "$OUTPUT_DIR" ] && OUTPUT_DIR="./output"
[ ! -n "$NGPU" ] && NGPU=1
[ ! -n "$BS" ] && BS=16
[ ! -n "$GRAD_ACC" ] && GRAD_ACC=1
[ ! -n "$EPOCHS" ] && EPOCHS=1500
[ "$AMP" == "true" ] && AMP_FLAG="--amp"

GBS=$(($NGPU * $BS * $GRAD_ACC))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}.\n"

echo -e "\nSetup: ${NGPU}x${BS}x${GRAD_ACC} - global batch size ${GBS}\n"

mkdir -p "$OUTPUT_DIR"
# python -m torch.distributed.launch --nproc_per_node ${NGPU} train.py \
python train.py \
    --cuda \
    --cudnn-enabled \
    -o "$OUTPUT_DIR/" \
    --log-file "$OUTPUT_DIR/nvlog.json" \
    --dataset-path Voituk_Narrative \
    --training-files filelists/Voituk_Narrative_mel_dur_pitch_train_filelist.txt \
    --validation-files filelists/Voituk_Narrative_mel_dur_pitch_valid_filelist.txt \
    --pitch-mean-std-file Voituk_Narrative/pitch_char_stats__Voituk_Narrative_train_filelist.json \
    --text-cleaners russian_cleaner \
    --epochs ${EPOCHS} \
    --epochs-per-checkpoint 100 \
    --warmup-steps 1000 \
    -lr 0.1 \
    -bs ${BS} \
    --optimizer lamb \
    --grad-clip-thresh 1000.0 \
    --dur-predictor-loss-scale 0.1 \
    --pitch-predictor-loss-scale 0.1 \
    --weight-decay 1e-6 \
    --gradient-accumulation-steps ${GRAD_ACC} \
    ${AMP_FLAG} \
    --resume
