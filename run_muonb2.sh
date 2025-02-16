eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh muon -z us-central2-b -t v4-256 --preemptible -- \
WANDB_API_KEY=1c85c63399be786e59026e288175122f49a434b0 \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_100M_muon.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/muonb/step10000_100M  \
--trainer.wandb.name 100M_muon_block_256_10k_respect \
--trainer.num_train_steps 10001 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.type muonb


