### GCN - strat 1 ###
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 20    --borf_batch_remove 10     --num_iterations 3    --dataset cora 
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 20    --borf_batch_remove 10     --num_iterations 3    --dataset citeseer 
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 30    --borf_batch_remove 10     --num_iterations 3    --dataset texas 
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 20    --borf_batch_remove 30     --num_iterations 2    --dataset cornell 
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 30    --borf_batch_remove 20     --num_iterations 2    --dataset wisconsin 

### GCN - strat 1 ###
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 2    --borf_batch_remove 1     --num_iterations 30    --dataset cora 
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 2    --borf_batch_remove 1     --num_iterations 30    --dataset citeseer 
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 3    --borf_batch_remove 1     --num_iterations 30    --dataset texas 
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 2    --borf_batch_remove 3     --num_iterations 20    --dataset cornell 
python compute_curvature_ranges.py     --rewiring borf     --layer_type GCN    --device cuda:0    --borf_batch_add 3    --borf_batch_remove 2     --num_iterations 20    --dataset wisconsin 
