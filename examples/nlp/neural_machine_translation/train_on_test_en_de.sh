python -m torch.distributed.launch --nproc_per_node $1 machine_translation_tutorial.py \
    --data_dir $2 \
    --dataset_name wmt14 \
    --src_lang en \
    --tgt_lang de \
    --num_layers 6 \
    --embedding_dropout 0.1 \
    --ffn_dropout 0.1 \
    --attn_score_dropout 0.1 \
    --attn_layer_dropout 0.1 \
    --warmup_steps 4000 \
    --tgt_tokenizer_model bpe_32k_ende_yttm.model \
    --src_tokenizer_model bpe_32k_ende_yttm.model \
    --optimizer adam \
    --max_steps 100000 \
    --batch_size 1024 
 
