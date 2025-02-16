eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh adam -z us-central2-b -t v4-512 --preemptible -- \
WANDB_API_KEY=1c85c63399be786e59026e288175122f49a434b0 \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_1b_adam.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/soap/lr4e-3_step10000_1b  \
--optimizer.learning_rate 4e-3 \
--optimizer.type soap \
--trainer.wandb.name soap_block_256_precond_10_10k_1b \
--optimizer.partition_grads_into_blocks True \
--optimizer.precondition_frequency 10 \
--optimizer.one_diag True \
--trainer.num_train_steps 10001 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.lr_schedule linear \
--optimizer.cooldown 0.0