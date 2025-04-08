eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh muon -z europe-west4-b -t v5litepod-128 --preemptible -- \
WANDB_API_KEY=1c85c63399be786e59026e288175122f49a434b0 \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_100M_sophia.yaml  \
--trainer.checkpointer.base_path "gs://marin-eu-west4/scratch/kaiyue/checkpoints/muonb/step10000_100M"  \
--trainer.wandb.name debug_sophia \
--trainer.num_train_steps 10001 \
--optimizer.min_lr_ratio 0.0 

