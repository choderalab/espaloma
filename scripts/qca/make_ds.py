import espaloma as esp
import pickle

def run():
    import os
    paths = os.listdir('merged_data')
    gs = [
        esp.Graph().load(
            'merged_data/' + path
        ) for path in paths
    ]
    
    ds = esp.data.dataset.GraphDataset(gs)

    ds.save('ds.th')

if __name__ == '__main__':
    run()
