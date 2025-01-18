eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh sophia4 -z us-central2-b -t v4-128 --preemptible -- \
WANDB_API_KEY=1c85c63399be786e59026e288175122f49a434b0 \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_100M_linear.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/wsds/sophiag/debug  \
--optimizer.type sophia-g \
--optimizer.learning_rate 6e-4 \
--trainer.wandb.name 100M-SophiaG-lr6e-4 