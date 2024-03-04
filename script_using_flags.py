# script_using_flags.py
from absl import app
from absl import flags
from sklearn.model_selection import ParameterGrid

flags.DEFINE_string('input_file', None, 'Path to the input file')
flags.DEFINE_integer('num_iterations', 10, 'Number of iterations')

params = ParameterGrid({"is_private": [0], "model_ntk":["cnn2d_1l", "fc_1l", "fc_2l"],"d_code": [11,21,31], "ntk_width": [30,50,100], "n_iter": [500, 1000, 2000, 4000, 8000, 16000, 25000, 32000], "lr": [0.01], "gen_batch": [200],"ntk_width_2": [100]})

# for p in params:
#     print("hyperparams: ", p)
#     sys.argv[4]=str(p['n_iter'])
#     sys.argv[6]=str(p['d_code'])
#     sys.argv[8]= str(p['ntk_width'])
#     sys.argv[10]=str(p['gen_batch'])
#     sys.argv[12]=str(p['lr'])
#     sys.argv[14]=str(p['ntk_width_2'])
#     sys.argv[16]=str(p['model_ntk'])

FLAGS = flags.FLAGS

def main(argv):
    print('Input File:', FLAGS.input_file)
    print('Number of Iterations:', FLAGS.num_iterations)
    return 0.5  # Replace with your actual return value

if __name__ == '__main__':
    app.run(main)
