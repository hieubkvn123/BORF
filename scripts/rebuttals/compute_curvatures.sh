### GCN ###
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 20    --borf_batch_remove 10     --num_iterations 3    --dataset cora 

python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 20    --borf_batch_remove 10     --num_iterations 3    --dataset citeseer 

python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 30    --borf_batch_remove 10     --num_iterations 3    --dataset texas 

python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 20    --borf_batch_remove 30     --num_iterations 2    --dataset cornell 

python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 30    --borf_batch_remove 20     --num_iterations 2    --dataset wisconsin 

python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 20    --borf_batch_remove 20     --num_iterations 3    --dataset chameleon 

