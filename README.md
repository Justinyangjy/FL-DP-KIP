# DP-KIP

## Dependendencies

    python 3.10.8
    absl-py 1.3.0  
    autodp 0.2
    cudatoolkit 11.3.1
    cudnn 8.2.1
    jax 0.4.1 
    jaxlib 0.4.1+cuda11.cudnn82
    matplotlib  3.6.2
    numpy  1.24.1
    scikit-learn 1.1.3 
    scipy 1.9.3 
    xgboost 1.7.3 
    tensorflow-datasets 4.8.1
    tensorflow 2.11.0
    seaborn 0.12.2
    pandas 1.5.3
    neural-tangents 0.6.1
    
  
## Run DP-KIP code

### Image data

 `python dpkip_inf_ntk.py --dpsgd=True --l2_norm_clip=1e-6 --epochs=10 --learning_rate=1e-2 --batch_size=50 --epsilon=1 --architecture='FC' --width=1024 --dataset='mnist' --support_size=10` 
 
### Tabular data

 `python dpkip_tab_data.py --dpsgd=True --reg=1e-6 --learning_rate=1e-1 --l2_norm_clip=1e-1 --batch_rate=0.01 --epochs=10 --dataset='credit' --undersampled_rate=0.01 --architecture='FC' --support_size=2 --width=1024`
