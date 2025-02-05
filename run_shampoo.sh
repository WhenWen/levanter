eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh shampoo -z us-central2-b -t v4-128 --preemptible -- \
WANDB_API_KEY=1c85c63399be786e59026e288175122f49a434b0 \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_100M_shampoo.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/shampoo/lr1e-3_step10000  \
--optimizer.learning_rate 1e-3 \
--optimizer.type shampoo 