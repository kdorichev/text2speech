#!/usr/bin/env bash

DATA_DIR="LJSpeech-1.1"
AMP="true" 

[ ! -n "$WAVEG_CH" ] && WAVEG_CH="pretrained_models/waveglow/waveglow_1076430_14000_amp.pt"
[ ! -n "$FASTPITCH_CH" ] && FASTPITCH_CH="pretrained_models/fastpitch/nvidia_fastpitch_200518.pt"
#"output/FastPitch_checkpoint_1500.pt"
[ ! -n "$BS" ] && BS=32
[ ! -n "$PHRASES" ] && PHRASES="phrases/devset10.tsv"
[ ! -n "$OUTPUT_DIR" ] && OUTPUT_DIR="./output/audio_$(basename ${PHRASES} .tsv)"
[ "$AMP" == "true" ] && AMP_FLAG="--amp"

python inference.py --cuda \
                    -i ${PHRASES} \
                    -o ${OUTPUT_DIR} \
                    --dataset-path ${DATA_DIR} \
                    --fastpitch ${FASTPITCH_CH} \
                    --waveglow ${WAVEG_CH} \
		    --wn-channels 256 \
                    --batch-size ${BS} \
                    ${AMP_FLAG}
