# Fed-DP-KIP

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
    
  
## Run Fed-DP-KIP

`/bin/python3 /home/justin/FedLearning/dpkip_inf_ntk_colored.py --dpsgd=True --epsilon=0.5 --width=100 --l2_norm_clip=1.0 --learning_rate=0.01 --batch_size=32 --reg=0.001 --epochs=10 --support_size=500 --result_images_path=result_images_0.5_500 > /home/justin/FedLearning/output_0.5_500/100_1.0_0.01_32_0.001_10.txt`

`/bin/python3 /home/justin/FedLearning/dpkip_inf_ntk_imbalance.py > /home/justin/FedLearning/output/Fed-DP-KIP_imblance.txt`

`/bin/python3 /home/justin/FedLearning/dpkip_inf_ntk.py > /home/justin/FedLearning/output/Fed-DP-KIP.txt`

