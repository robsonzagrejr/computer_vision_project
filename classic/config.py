
img_size = (480,852)

seed = 72
data_path = 'data/data.pkl'
params_path= 'data/feature/best_param.json'
model_path = 'data/model/hog_sgd_model.pkl'
x_scalar_path = 'data/feature/x_scaler.pkl'

windows_chunks_definition = {
    'big': {
        'scale': 3.5,
        'bottom': img_size[0]*1,
        'color': (255,0,0),
        'line_size': 3
    },
    'middle': {
        'scale': 3,
        'bottom': img_size[0]*0.85,
        'color': (0,0,255),
        'line_size': 2
    },
    'small': {
        'scale': 2,
        'bottom': img_size[0]*0.7,
        'color': (0,255,0),
        'line_size': 2
    },
    'ssmall': {
        'scale': 1,
        'bottom': img_size[0]*0.65,
        'color': (0,0,255),
        'line_size': 2
    },
    'sssmall': {
        'scale': 0.75,
        'bottom': img_size[0]*0.5,
        'color': (255,0,255),
        'line_size': 2
    },
}