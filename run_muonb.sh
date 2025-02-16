eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh muadambase -z us-central2-b -t v4-256 --preemptible -- \
WANDB_API_KEY=1c85c63399be786e59026e288175122f49a434b0 \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_100M_muon.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/muadam4/step50000_100M  \
--trainer.wandb.name 100M_muon_adam_lr_grafting_50k \
--trainer.num_train_steps 50001 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.lr_schedule linear \
--optimizer.type muadam4 \
--optimizer.cooldown 0.0 \
--optimizer.max_grad_norm 0.0