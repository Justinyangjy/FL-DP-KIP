import os
import subprocess
from sklearn.model_selection import ParameterGrid

# Set the parameter values you want to pass


private_params = ParameterGrid({"is_private": [True], "epsilon": [10], "learning_rate":[1e-01, 1e-02, 1e-03, 1e-04],
                                "reg": [1e-01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07], "epochs": [10,20,50],
                                  "batch_size": [64, 128, 256], "width": [1000], "l2_norm_clip": [1e-03, 1e-04, 1e-05, 1e-06]})

cur_path = os.path.dirname(os.path.abspath(__file__))
best_path = os.path.join(cur_path, 'best')
if not os.path.exists(best_path):
    os.makedirs(best_path)

acc_best = 0
acc_best_file_name = ''

for p in private_params:
    is_private= p['is_private']
    epsilon = p['epsilon']
    width = p['width']
    l2_norm_clip = p['l2_norm_clip']
    learning_rate = p['learning_rate']
    batch_size = p['batch_size']
    reg = p['reg']
    epochs = p['epochs']
    support_size = 10

    save_path = os.path.join(cur_path, f'output_{epsilon}_{support_size}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    command = [
        '/bin/python3',
        '/home/justin/FedLearning/dpkip_inf_ntk_colored.py',
        f'--dpsgd={is_private}',
        f'--epsilon={epsilon}',
        f'--width={width}',
        f'--l2_norm_clip={l2_norm_clip}',
        f'--learning_rate={learning_rate}',
        f'--batch_size={batch_size}',
        f'--reg={reg}',
        f'--epochs={epochs}',
        f'--support_size={support_size}',
        f'--result_images_path=result_images_{epsilon}_{support_size}',
        ">", f"/home/justin/FedLearning/output_{epsilon}_{support_size}/${width}_${l2_norm_clip}_${learning_rate}_${batch_size}_${reg}_${epochs}.txt "
        ]
    
    # Use subprocess.run() to run the command and capture the returncode
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return_code = float("".join(result.stderr.splitlines()[-1:]))
        output = result.stdout

        print(f'Acc: {return_code}')
        # print(f'Output: {output}')
        if acc_best < return_code:
            acc_best = return_code
            acc_best_file_name = f"${width}_${l2_norm_clip}_${learning_rate}_${batch_size}_${reg}_${epochs}"
    except Exception as e:
        print("An unexpected error occurred:", str(e))


print('acc_best_file_name: ', acc_best_file_name)
print(f'acc_best: {acc_best}')

with open(os.path.join(best_path, "best.txt"), 'a') as f:
    f.writelines(str(acc_best)+"\n")
    f.writelines(str(acc_best_file_name)+"\n")



private_params = ParameterGrid({"is_private": [True], "epsilon": [10], "learning_rate":[1e-01, 1e-02, 1e-03, 1e-04],
                                "reg": [1e-01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07], "epochs": [10,20,50],
                                  "batch_size": [64, 128, 256], "width": [1000], "l2_norm_clip": [1e-03, 1e-04, 1e-05, 1e-06]})

acc_best = 0
acc_best_file_name = ''

for p in private_params:
    is_private= p['is_private']
    epsilon = p['epsilon']
    width = p['width']
    l2_norm_clip = p['l2_norm_clip']
    learning_rate = p['learning_rate']
    batch_size = p['batch_size']
    reg = p['reg']
    epochs = p['epochs']
    support_size = 100

    save_path = os.path.join(cur_path, f'output_{epsilon}_{support_size}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    command = [
        '/bin/python3',
        '/home/justin/FedLearning/dpkip_inf_ntk_colored.py',
        f'--dpsgd={is_private}',
        f'--epsilon={epsilon}',
        f'--width={width}',
        f'--l2_norm_clip={l2_norm_clip}',
        f'--learning_rate={learning_rate}',
        f'--batch_size={batch_size}',
        f'--reg={reg}',
        f'--epochs={epochs}',
        f'--support_size={support_size}',
        f'--result_images_path=result_images_{epsilon}_{support_size}',
        ">", f"/home/justin/FedLearning/output_{epsilon}_{support_size}/${width}_${l2_norm_clip}_${learning_rate}_${batch_size}_${reg}_${epochs}.txt "
        ]
    
    # Use subprocess.run() to run the command and capture the returncode
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return_code = float("".join(result.stderr.splitlines()[-1:]))
        output = result.stdout

        print(f'Acc: {return_code}')
        # print(f'Output: {output}')
        if acc_best < return_code:
            acc_best = return_code
            acc_best_file_name = f"${width}_${l2_norm_clip}_${learning_rate}_${batch_size}_${reg}_${epochs}"
    except Exception as e:
        print("An unexpected error occurred:", str(e))




with open(os.path.join(best_path, "best.txt"), 'a') as f:
    f.writelines(str(acc_best)+"\n")
    f.writelines(str(acc_best_file_name)+"\n")

private_params = ParameterGrid({"is_private": [True], "epsilon": [10], "learning_rate":[1e-01, 1e-02, 1e-03, 1e-04],
                                "reg": [1e-01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07], "epochs": [10,20,50],
                                  "batches": [64, 128, 256], "width": [1000], "l2_norm_clip": [1e-03, 1e-04, 1e-05, 1e-06]})

acc_best = 0
acc_best_file_name = ''

for p in private_params:
    is_private= p['is_private']
    epsilon = p['epsilon']
    width = p['width']
    l2_norm_clip = p['l2_norm_clip']
    learning_rate = p['learning_rate']
    batch_size = p['batch_size']
    reg = p['reg']
    epochs = p['epochs']
    support_size = 500

    save_path = os.path.join(cur_path, f'output_{epsilon}_{support_size}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    command = [
        '/bin/python3',
        '/home/justin/FedLearning/dpkip_inf_ntk_colored.py',
        f'--dpsgd={is_private}',
        f'--epsilon={epsilon}',
        f'--width={width}',
        f'--l2_norm_clip={l2_norm_clip}',
        f'--learning_rate={learning_rate}',
        f'--batch_size={batch_size}',
        f'--reg={reg}',
        f'--epochs={epochs}',
        f'--support_size={support_size}',
        f'--result_images_path=result_images_{epsilon}_{support_size}',
        ">", f"/home/justin/FedLearning/output_{epsilon}_{support_size}/${width}_${l2_norm_clip}_${learning_rate}_${batch_size}_${reg}_${epochs}.txt "
        ]
    
    # Use subprocess.run() to run the command and capture the returncode
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return_code = float("".join(result.stderr.splitlines()[-1:]))
        output = result.stdout

        print(f'Acc: {return_code}')
        # print(f'Output: {output}')
        if acc_best < return_code:
            acc_best = return_code
            acc_best_file_name = f"${width}_${l2_norm_clip}_${learning_rate}_${batch_size}_${reg}_${epochs}"
    except Exception as e:
        print("An unexpected error occurred:", str(e))




print('acc_best_file_name: ', acc_best_file_name)
print(f'acc_best: {acc_best}')

with open(os.path.join(best_path, "best.txt"), 'a') as f:
    f.writelines(str(acc_best)+"\n")
    f.writelines(str(acc_best_file_name)+"\n")