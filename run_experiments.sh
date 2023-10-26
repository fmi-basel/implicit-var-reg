# byol cifar10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name byol.yaml                      \
    data.dataset=cifar10                         \
    name=byol_cifar10 &

# byol cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name byol.yaml                      \
    data.dataset=cifar100                        \
    name=byol_cifar100 &

# byol stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name byol.yaml                      \
    data.dataset=stl10                           \
    name=byol_stl10 &

# byol tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name byol.yaml                      \
    data.dataset=tinyimagenet                    \
    name=byol_tinyimagenet &





# simsiam (byol without momentum encoder) cifar10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name byol-noema.yaml                \
    data.dataset=cifar10                         \
    name=simsiam_cifar10 &

# simsiam cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name byol-noema.yaml                \
    data.dataset=cifar100                        \
    name=simsiam_cifar100 &

# simsiam stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name byol-noema.yaml                \
    data.dataset=stl10                           \
    name=simsiam_stl10 &

# simsiam tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name byol-noema.yaml                \
    data.dataset=tinyimagenet                    \
    name=simsiam_tinyimagenet &





# directpred cifar10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name directpred.yaml                \
    data.dataset=cifar10                         \
    name=directpred_cifar10                      \
    optimizer.weight_decay=1e-4 &

# directpred cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name directpred.yaml                \
    data.dataset=cifar100                        \
    name=directpred_cifar100                     \
    optimizer.weight_decay=1e-4 &

# directpred stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name directpred.yaml                \
    data.dataset=stl10                           \
    name=directpred_stl10                        \
    optimizer.weight_decay=1e-4 &

# directpred tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name directpred.yaml                \
    data.dataset=tinyimagenet                    \
    name=directpred_tinyimagenet                 \
    optimizer.weight_decay=1e-4 &





# directpred (without momentum encoder) cifar10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name directpred-noema.yaml          \
    data.dataset=cifar10                         \
    name=directpred-noema_cifar10                \
    optimizer.weight_decay=1e-4 &

# directpred (without momentum encoder) cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name directpred-noema.yaml          \
    data.dataset=cifar100                        \
    name=directpred-noema_cifar100               \
    optimizer.weight_decay=1e-4 &

# directpred (without momentum encoder) stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name directpred-noema.yaml          \
    data.dataset=stl10                           \
    name=directpred-noema_stl10                  \
    optimizer.weight_decay=1e-4 &

# directpred (without momentum encoder) tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name directpred-noema.yaml          \
    data.dataset=tinyimagenet                    \
    name=directpred-noema_tinyimagenet           \
    optimizer.weight_decay=1e-4 &





# isoloss cifar10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name isoloss.yaml                   \
    data.dataset=cifar10                         \
    name=isoloss_cifar10                         \
    optimizer.weight_decay=1e-4 &

# isoloss cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name isoloss.yaml                   \
    data.dataset=cifar100                        \
    name=isoloss_cifar100                        \
    optimizer.weight_decay=1e-4 &

# isoloss stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name isoloss.yaml                   \
    data.dataset=stl10                           \
    name=isoloss_stl10                           \
    optimizer.weight_decay=1e-4 &

# isoloss tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name isoloss.yaml                   \
    data.dataset=tinyimagenet                    \
    name=isoloss_tinyimagenet                    \
    optimizer.weight_decay=1e-4 &





# isoloss (without momentum encoder) cifar10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name isoloss-noema.yaml             \
    data.dataset=cifar10                         \
    name=isoloss-noema_cifar10                   \
    optimizer.weight_decay=1e-4 &

# isoloss (without momentum encoder) cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name isoloss-noema.yaml             \
    data.dataset=cifar100                        \
    name=isoloss-noema_cifar100                  \
    optimizer.weight_decay=1e-4 &

# isoloss (without momentum encoder) stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name isoloss-noema.yaml             \
    data.dataset=stl10                           \
    name=isoloss-noema_stl10                     \
    optimizer.weight_decay=1e-4 &

# isoloss (without momentum encoder) tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name isoloss-noema.yaml             \
    data.dataset=tinyimagenet                    \
    name=isoloss-noema_tinyimagenet              \
    optimizer.weight_decay=1e-4 &





# directcopy cifar10
python3 main_pretrain.py                         \
     --config-path scripts/pretrain/cifar/       \
    --config-name directpred.yaml                \
    data.dataset=cifar10                         \
    name=directcopy_cifar10                      \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name directpred.yaml                \
    data.dataset=cifar100                        \
    name=directcopy_cifar100                     \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name directpred.yaml                \
    data.dataset=stl10                           \
    name=directcopy_stl10                        \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name directpred.yaml                \
    data.dataset=tinyimagenet                    \
    name=directcopy_tinyimagenet                 \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &





# directcopy (without momentum encoder) cifar10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name directpred-noema.yaml          \
    data.dataset=cifar10                         \
    name=directcopy-noema_cifar10                \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy (without momentum encoder) cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name directpred-noema.yaml          \
    data.dataset=cifar100                        \
    name=directcopy-noema_cifar100               \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy (without momentum encoder) stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name directpred-noema.yaml          \
    data.dataset=stl10                           \
    name=directcopy-noema_stl10                  \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy (without momentum encoder) tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name directpred-noema.yaml          \
    data.dataset=tinyimagenet                    \
    name=directcopy-noema_tinyimagenet           \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &




# directcopy (isoloss) cifar10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name isoloss.yaml                   \
    data.dataset=cifar10                         \
    name=directcopy-isocopy_cifar10              \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy (isoloss) cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name isoloss.yaml                   \
    data.dataset=cifar100                        \
    name=directcopy-isocopy_cifar100             \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy (isoloss) stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name isoloss.yaml                   \
    data.dataset=stl10                           \
    name=directcopy-isocopy_stl10                \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy (isoloss) tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name isoloss.yaml                   \
    data.dataset=tinyimagenet                    \
    name=directcopy-isocopy_tinyimagenet         \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &




# directcopy (isoloss, without momentum encoder) cifar10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name isoloss-noema.yaml             \
    data.dataset=cifar10                         \
    name=directcopy-isocopy-noema_cifar10        \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy (isoloss, without momentum encoder) cifar100
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/cifar/        \
    --config-name isoloss-noema.yaml             \
    data.dataset=cifar100                        \
    name=directcopy-isocopy-noema_cifar100       \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy (isoloss, without momentum encoder) stl10
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/stl-10/       \
    --config-name isoloss-noema.yaml             \
    data.dataset=stl10                           \
    name=directcopy-isocopy-noema_stl10          \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &

# directcopy (isoloss, without momentum encoder) tinyimagenet
python3 main_pretrain.py                         \
    --config-path scripts/pretrain/tinyimagenet/ \
    --config-name isoloss-noema.yaml             \
    data.dataset=tinyimagenet                    \
    name=directcopy-isocopy-noema_tinyimagenet   \
    optimizer.weight_decay=1e-4                  \
    method_kwargs.dp_alpha=1                     \
    method_kwargs.eps_iso=0.25 &