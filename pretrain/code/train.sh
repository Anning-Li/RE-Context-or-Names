python -m torch.distributed.launch --nproc_per_node 4 \ 
	CUDA_VISIBLE_DEVICES=4,5,6,7 main.py \
	--cuda 4,5,6,7 \
        --model MTB \
	--lr 3e-5 --batch_size_per_gpu 32 --max_epoch 20 \
	--gradient_accumulation_steps 2 \
	--max_length 64 \
	--save_step 5000 \
	--alpha 0.3 \
	--temperature 0.05 \
	--train_sample \
	--save_dir ckpt_mtb \
