dataset=dynamic
backbone=HFEbackbone
predictor=BHSTPredictor
loss_fn=a
npts=46
topk=5
device=0
python main.py --train --resume --visualize --loss_fn ${loss_fn} --device ${device} --num_kpts ${npts} --top_k ${topk} --sample_rate 1.0 --engine tem_seq \
--backbone_name ${backbone} \
--predictor_name ${predictor} \
--cfg_path lib/config/config_sgd.yaml \
