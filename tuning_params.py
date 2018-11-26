import pandas as pd
import numpy as np
from lightfm import LightFM
import time
from lightfm.evaluation import auc_score
from skopt import forest_minimize


def objective(params):
    # unpack
    epochs, learning_rate, no_components = params

    model = LightFM(loss='warp',
                    random_state=2016,
                    learning_rate=learning_rate,
                    no_components=no_components)
    model.fit(train, epochs=epochs,
              num_threads=4, verbose=True)

    patks = auc_score(model, test, num_threads=4)
    maptk = np.mean(patks)
    # Make negative because we want to _minimize_ objective
    out = -maptk
    # Handle some weird numerical shit going on
    if np.abs(out + 1) < 0.01 or out < -1.0:
        return 0.0
    else:
        return out



space = [(1, 40), # epochs
         (10**-4, 0.5, 'log-uniform'), # learning_rate
         (20, 200), # no_components
        ]

res_fm = forest_minimize(objective, space, n_calls=20,
                     random_state=0,
                     verbose=True)
print('Maximimum auc found: {:6.5f}'.format(-res_fm.fun))
print('Optimal parameters:')
params = ['epochs', 'learning_rate', 'no_components']
for (p, x_) in zip(params, res_fm.x):
    print('{}: {}'.format(p, x_))